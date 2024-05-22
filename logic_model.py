import os
import math
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler
from utils.data_utils import (
    get_Xt,
)

from utils.logic_utils import (
    get_logic_features, rules_mapping
)
TOLERANCE = 1e-5


class CosineAnnealingWithExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, gamma=0.99):
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma
        super(CosineAnnealingWithExponentialDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 * 
            (self.gamma ** self.last_epoch)
            for base_lr in self.base_lrs
        ]

class LogicModel(nn.Module):

    def __init__(
            self, 
            head_predicates,
            body_predicates,
            rule_max_length = 3,
        ):
        super().__init__()
        self.epsilon = 1e-5
        self.head_predicates = head_predicates
        self.body_predicates = body_predicates

  
        self.head_pred_nums = len(head_predicates)
        self.body_pred_nums = len(body_predicates)
        self.pred_nums = self.head_pred_nums + self.body_pred_nums


        self.predicates = [i for i in range(self.head_pred_nums+self.body_pred_nums)]

        self.rule_mapping, self.inverse_mapping = rules_mapping(
                    head_predicates=head_predicates, 
                    complete_predicates=self.predicates,
                    max_len= rule_max_length
                )
        self.rule_space = list(self.rule_mapping.values())
        self.rule_weights = nn.Parameter(data = torch.zeros(size=(len(self.rule_mapping),)))
        self.base_weights = nn.Parameter(data = torch.zeros(
                                    size=(self.head_pred_nums+self.body_pred_nums, 
                                          )))
        self.gamma = nn.Parameter(data = torch.ones(size=(self.head_pred_nums * len(self.rule_mapping),)))
        self.beta = nn.Parameter(data = torch.zeros(size=(self.head_pred_nums * len(self.rule_mapping),)))

        # Initialize variables to track mean and variance
        self.register_buffer('running_mean', torch.zeros(size=(self.head_pred_nums * len(self.rule_mapping),)))
        self.register_buffer('running_var', torch.ones(size=(self.head_pred_nums * len(self.rule_mapping),)))
        self.register_buffer('M2', torch.zeros(size=(self.head_pred_nums * len(self.rule_mapping),)))
        self.register_buffer('n', torch.tensor(0.0))
        
    def restore_gt_rules(
            self,
            gt_rules,
            gt_rules_weights,
            gt_body_mu,
            gt_head_mu,
    ):
        """Restore the gt rule weights and base weights"""
        gt_rules_idx = [self.rule_mapping[tuple(r)] for r in gt_rules]
        self.rule_weights.data[gt_rules_idx] = torch.Tensor(gt_rules_weights).to(self.rule_weights)
        self.base_weights.data[self.head_predicates] = torch.Tensor(gt_head_mu).to(self.rule_weights)
        self.base_weights.data[self.body_predicates] = torch.Tensor(gt_body_mu).to(self.rule_weights)
    

    def update_running_stats(self, x):
        """Welford's algorithm to calculate the variance of streaming data using second moment"""
        # Incremental update of mean and second moment (M2)

        self.n += 1
        delta = x - self.running_mean
        self.running_mean += delta / self.n
        delta2 = x - self.running_mean
        self.M2 += delta * delta2

        # Update running variance
        if self.n > 1:
            self.running_var = self.M2 / (self.n - 1)
        else:
            # For the first sample, set running variance to 1
            self.running_var = torch.ones_like(self.running_mean)

    def feature_normalization(self, x):
        if self.training:
            # Update online mean and variance
            self.update_running_stats(x)

            # Normalize input
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        else:
            # Use running mean during inference
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

        # Apply scaling and shifting
        y = x_normalized * self.gamma + self.beta
        return y


    def get_intensity(self, t, X, predicates, rules) -> torch.tensor:
        """Return the intensity function given event histories before/after time t"""
        X = get_Xt(X,t)
        ## recursively calculate the logic features
        logic_features = get_logic_features(X, predicates, rules, self.rule_mapping)
        logic_features = torch.Tensor(logic_features).to(self.rule_weights)

        ## features normalization
        logic_features = torch.ravel(logic_features)
        logic_features = self.feature_normalization(logic_features)
        logic_features = logic_features.view(1, self.head_pred_nums, len(self.rule_mapping))
        # print(rules, logic_features.shape)
        intensity = F.softplus(
            torch.matmul(logic_features, self.rule_weights) + self.base_weights[predicates], 
            beta=1
            )
        # print("base", self.base_weights)
        # print("weights_grad", self.rule_weights.grad)
        # print("intensity", intensity)
        return intensity

    
    def intensity_log_sum(self, X : list[list], predicates : list, rules: list) -> torch.tensor:
        """Return the log-likelihood at event time"""
        log_intensity = 0
        # sum over batch dim
        for sample in X:
            # sume over time dim
            for i in range(1, len(sample)+1):
                cur_h, cur_t = sample[:i, :], sample[i-1, 0]
                intensity = self.get_intensity(
                        cur_t, [cur_h], predicates, rules
                        )
                log_intensity += torch.log(torch.sum(intensity, dim=-1))  ## sum over type dim -> [1, ]

        log_intensity = log_intensity[0] / len(X) ## mean over sample dim
        return log_intensity

    def intensity_integral(self, X : list[list], end_time, predicates, rules, linspace=200) -> torch.tensor:
        """Return the log-likelihood at non-event time by MC integation"""
        start_time = 0
        intensity_sum = 0
        for t in np.linspace(start_time, end_time, linspace):
            cur_intensity = self.get_intensity(t, X, predicates, rules)
            cur_intensity = torch.sum(cur_intensity, dim=-1)       # sum over type dim
            intensity_sum += cur_intensity # [Bs, ]
        integral = intensity_sum * (end_time - start_time) # MC Integration
        integral = torch.mean(integral, dim=0) # mean over sample dim
        return integral

    def regularization(self):
        return 10 * torch.norm(self.base_weights) + torch.norm(self.rule_weights)

    def log_likelihood(self, t : float, X : list, predicates : list, rules : list, linspace=100) -> torch.tensor:
        """Return the llh of a point process"""
        event_llh = self.intensity_log_sum(X=X, predicates=predicates, rules=rules)
        non_event_llh = self.intensity_integral(X, t, predicates, rules, linspace)
        return event_llh - non_event_llh
    
    def cross_entropy_loss(self, t : float, X : list, y : np.ndarray, predicates : list) -> torch.tensor:
        """Return cross entropy loss of next event"""
        y_one_hot = F.one_hot(torch.LongTensor(y), num_classes=self.pred_nums).to(self.base_weights)
        y_one_hot = y_one_hot[:, predicates]
        pred_prob = self.pred_prob(t, X, predicates)

        loss = F.cross_entropy(input=pred_prob, target=y_one_hot)

        return loss 
    
    def pred_prob(self, t, X, predicates) -> torch.tensor:
        """Return the probability of next event"""
        intensity = self.get_intensity(t, X, predicates)
        return intensity / intensity.sum(dim=-1, keepdim=True)
    

class LogicModelTrainer:
    def __init__(
            self,
            model     : LogicModel,
            train_params : dict = {},
            data_params : dict = {},
    ) -> None:
        
        self.model = model
        self.train_params = train_params
        self.data_params  = data_params

        # self.horizon = self.data_params.get('horizon', 5) 
        self.init_opt()

    def init_opt(
            self, 
        ):

        self.opt = torch.optim.SGD(
                params=self.model.parameters(),
                # line_search_fn="strong_wolfe"
                lr = self.train_params.get('lr', 0.01),
                )
        # self.lr_sched_exp = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer = self.opt,
        #     gamma= self.train_params.get('gamma_lr_decay',0.1)
        # )

        # self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer = self.opt,
        #     T_max= self.train_params.get('T_max', 20),
        #     eta_min = 1e-4
        # )

        self.lr_sched = CosineAnnealingWithExponentialDecay(
            optimizer = self.opt,
            T_max= self.train_params.get('T_max', 20),
            eta_min = 1e-5,
            gamma= self.train_params.get('gamma_lr_decay',0.995)
        )

        
    def train_one_step(
            self,
            batch_x,
            rules,
    ):
        # self.opt.zero_grad()
        t = np.max([x[-1,0] for x in batch_x])+1
        nll = -self.model.log_likelihood(
            t=t, 
            # t=self.horizon,
            X=batch_x, 
            predicates=self.model.head_predicates,
            rules= rules,
        )
        loss = nll + self.model.regularization()
        # loss = logic_model.cross_entropy_loss(t=horizon, X=batch_x, y=batch_y, predicates=head_predicates)
        loss.backward()
        # self.opt.step()

        # Check if loss is NaN
        if torch.isnan(loss):

            # Print relevant variables
            print("Model Weights:", self.model.state_dict())
            print("Gradients:", [p.grad for p in self.model.parameters()])
            print("Data Sample:", batch_x)
            print("Rule", self.model.rules)
            raise 

        return nll
    
    def save_rule_weights(
            self,
            path,
            name = ''
    ):
        rules_weights_np = self.model.rule_weights.data.cpu().numpy()
        res = {}
        for i in range(len(rules_weights_np)):
            res[str(self.model.inverse_mapping[i])] = float(rules_weights_np[i])
        
        with open(os.path.join(path, f'rule_weights_{name}.json'), 'w') as f:
            json.dump(res, f)

    def save_weights(
            self,
            path,
    ):
        torch.save(self.model.state_dict(), path)
    
    def lr_sched_step(
            self
    ):
        self.lr_sched.step()
        
        # self.lr_sched_exp.step()
    
    def train(
            self,
            batch_x,
            steps,
            rules,
    ):  
        step_loss = 0
        prev_loss = 0
        for i in range(steps):
            loss = self.train_one_step(batch_x, rules)
            loss = loss.item()
            # Check for convergence
            # if abs(prev_loss - loss) < TOLERANCE:
            #     print(f"Converged after {i+1} iterations")
            #     break
            prev_loss = loss
            step_loss += loss
            del loss

        return step_loss / steps
