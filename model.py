from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import pdb

from utils.llm_utils import (
    get_target_tokens_logp,
    get_target_y,
    propose_prompt_wrap,
    sol_prompt_wrap,
    state_to_rules,
    get_path_logprob
)

from utils.train_utils import (
    calc_rank
)

from typing import Union
from utils.prompt_template import (
    llama_prompt,
    prompt_thoughts_example,
    prompt_thought_prior,
    prompt_thoughts,
    prompt_sol,
    prompt_observation,
)

from utils.prompt_examples import (
    example_0, example_1, example_2
)

from logic_model import LogicModelTrainer, LogicModel


def db_loss(
        log_rewards  : torch.tensor,
        log_pf : torch.tensor,
        log_pterm : torch.tensor
) -> torch.tensor:
    # p(s2 | s1) r(s1) p(T| s2) = p (s1 | s2) r(s2) p(T s1)
    # assume backward prob p (s1 | s2) == 1
    return (log_rewards[:-1] + log_pf[:-1] + log_pterm[1:] \
              - log_rewards[1:] - log_pterm[:-1]) ** 2


def subtb_loss(
        logrewards,
        logPF,
        eos_logprob,
):
    """modified subTB loss with logpb=0"""
    subtb_lambda = 1.
    max_len = logrewards.size(-1)
    delta = logrewards[:-1] - eos_logprob[:-1] + logPF[:-1] - (logrewards[1:] - eos_logprob[1:])

    delta_cumsum = torch.cat( [ torch.zeros_like(delta[ :1]), delta ], -1).cumsum(-1)
    
    # get trajectory lengths by summing the mask
    batch_loss = 0.
    total_lambda = 0.
    for subtraj_len in range(1, max_len+1):
        subtb_term = (delta_cumsum[subtraj_len:] - delta_cumsum[:-subtraj_len])**2
        
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += subtb_lambda ** (subtraj_len - 1)
    batch_loss /= total_lambda

    return batch_loss

def get_tree_mask(
        rest_path : list[str],
        current_path : list[str],
        current_state : list[list[str]]
):
    """Return the pobability mask of a tree
    Example 
        rest_path = ["C", "D"]
        current_path = ["A"]
        current_state = [["A","C"]]
    Return mask : [True, False]
    """
    mask = torch.zeros(len(rest_path))

    for i, path in enumerate(rest_path):
        tmp = current_path.copy()
        tmp.append(path)
        # if the path already exist in the tree
        if tmp in current_state:
            mask[i] = 1
    return mask

def apply_tree_mask(
        mask : torch.tensor,
        logits : torch.tensor
):
    masked_logits = logits + (mask * -1e9)
    return masked_logits


def combine_arrays(arr1, idx1, arr2, idx2):
    # Determine the size of the new array
    max_index = len(arr1) + len(arr2)
    combined = [None] * (max_index)

    # Insert elements from the first array
    for i, index in enumerate(idx1):
        combined[index] = arr1[i]

    # Insert elements from the second array
    for i, index in enumerate(idx2):
        combined[index] = arr2[i]

    # Handling empty slots (if any) - here, we simply leave them as None
    return combined


# Define a dummy context manager to use when not in inference mode
class dummy_context_manager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class EventGPTGFN:

    def __init__(
            self,
            generation_model : callable, 
            generation_model_tokenizer,
            inference_model,
            inference_model_tokenizer,
            prior_model : callable,
            prior_model_tokenizer,
            knowledge_model : LogicModel,
            head_preds : dict,
            body_preds : dict,
            max_depth : int = 5,
            max_width : int = 3,
            topk : int = 2,
            epsilon : float = 0.1,
            learning_prior : bool = False,
            example_nums = 0
    ):
        assert topk <= max_width
        self.generation_model = generation_model
        self.knowledge_model = knowledge_model
        self.inference_model = inference_model
        self.inference_model_tokenizer = inference_model_tokenizer
        self.prior_model = prior_model
        self.prior_model_tokenizer = prior_model_tokenizer
        self.generation_model_tokenizer = generation_model_tokenizer
        self.head_preds = list(head_preds.values())
        self.body_preds = list(body_preds.values())
        
        self.head_preds_idx = list(head_preds.keys())
        self.body_preds_idx = list(body_preds.keys())
        # print(self.body_preds_idx)
        self.candidate_preds = combine_arrays(
            arr1=self.head_preds,
            idx1=self.head_preds_idx,
            arr2=self.body_preds,
            idx2=self.body_preds_idx
        )

        self.epsilon = epsilon
        self.max_depth = max_depth
        self.max_width = max_width
        self.top_k = topk
        self.learning_prior = learning_prior
        
        examples = [
            example_0,
            example_1,
            example_2
        ]
        if example_nums > 0:
            self.examples = '\n'.join(examples[:example_nums])
        else:
            self.examples = '' 
        # self.init_state = [["4"]]
    
    def eval(
            self,
            history : np.ndarray,
            target_y : int,
    ):
        """ evaluate the performance in inference model"""
        self.generation_model.eval()
        self.prior_model.eval()
        done = False
        truncated = False
        current_state = [[self.candidate_preds[target_y]]] ## init current state to be label
        entropy_list = []
        cnt = 0
        while not (done or truncated):
            (
            next_state, _, 
            _, _, llh,
            pred_info, done, info
            ) = self.step(
                history, 
                target_y, 
                current_state,
                eval=True
            )
            cnt += 1
            truncated = cnt >=self.max_depth
            current_state = next_state
            entropy = 0
            for path in next_state:
                logp = get_path_logprob(
                    self.generation_model,
                    self.generation_model_tokenizer,
                    history,
                    path,
                    self.candidate_preds,
                    device=self.generation_model.device
                )
                entropy = entropy - (logp * torch.exp(logp))
                # print(entropy)
            entropy_list.append(entropy)

            # print(next_state)
        pred_info['entropy_hist'] = entropy_list
        return llh, pred_info
    

    def act(
            self,
            history : np.ndarray,
            target_y: int,
    ):  
        """Sample the thoughts in one episode"""
        self.generation_model.train()
        self.prior_model.train()
        logrewards = []
        logeosprobs = []
        logpf = []
        preds = []
        # initialize the state by target
        current_state = [[self.candidate_preds[target_y]]]

        # random start a state
        # current_state = [random.sample(self.head_preds,k=1)]
        curr_state_history = []
        next_state_history = []
        rest_state_history = []
        action_history = []
        hist_info = {}
        done = False

        ## reward for s_{0}
        # print("target_y", target_y)
        # log_reward, llh, pred_info = self.logreward(
        #     history=history,
        #     target_y= target_y, 
        #     state = current_state,
        #     logic_model=self.knowledge_model,
            
        # )
        # logrewards.append(log_reward)
        cnt = 0
        truncated = False
        while not (done or truncated):
            (
            next_state, next_state_logprob, 
            end_state_logprob, log_reward, llh,
            pred_info, done, info
            ) = self.step(history, target_y, current_state)
            cnt += 1

            truncated = cnt >=self.max_depth

            # if not (done or truncated):
            #     # reward s_{t}
            logrewards.append(log_reward)
            logeosprobs.append(end_state_logprob.sum())
            logpf.append(next_state_logprob.sum())
            preds.append(pred_info)      # append prediction
        
            next_state_history.append(next_state)
            curr_state_history.append(current_state)
            rest_state_history.append(info['rest_state'])
            action_history.append(info['actions'])
            current_state = next_state
            
        # [len + 1]
        logrewards = torch.stack(logrewards, dim=-1)
        # [len]
        logeosprobs = torch.stack(logeosprobs, dim=-1)
        logpf = torch.stack(logpf, dim=-1)
        # print('logrewards', logrewards)
        # print('logeosprob', logeosprobs)
        # print('logpf', logpf)
        hist_info['next_state_hist'] = next_state_history
        hist_info['curr_state_hist'] = curr_state_history
        hist_info['rest_state_hist'] = rest_state_history
        hist_info['actions_hist'] = action_history
        return (
            current_state, logrewards, 
            logeosprobs, logpf, llh, preds, hist_info
        )

    def step(
            self,
            history : np.ndarray,
            target_y : int, 
            current_state : list[list[str]],
            eval : bool = False,
    ):
        """ One step forward """
        with torch.inference_mode() if eval else dummy_context_manager():
            (
                next_state,  next_state_logprob, 
                end_state_logprob, done, info
            ) = self.generate_next_state_logprob(
                history,
                current_state,
            )
            # print(current_state)
            # print(next_state)
            # pdb.set_trace()
            log_reward, llh, pred_info = self.logreward(
                history=history,
                target_y= target_y, 
                state = next_state,
                logic_model=self.knowledge_model,
            )

        return (
            next_state, next_state_logprob, 
            end_state_logprob, log_reward, llh, pred_info, done, info
        )

    def get_rest_path(
            self,
            current_path : list[str] 
    ) -> list[str]:
        """Return the remaining path given the path already taken"""
        rest_path = self.candidate_preds.copy()
        # print(self.candidate_preds)
        # print(current_path)
        [rest_path.remove(item) for item in current_path]

        ## sample-muzero
        rest_path = np.random.choice(
            rest_path,
            size = min(len(rest_path), self.max_width),
            replace=False,
        )
        return rest_path
         
    def next_path_logprob(
            self,
            history : np.ndarray,
            current_path : list[str],
            rest_path : list[str],
    ) -> torch.tensor:
        """ Return the logprob of next path given current path """
        thoughts_query = propose_prompt_wrap(
                history, 
                y_list =current_path,
                rest_event=rest_path,
                candidate_preds=self.candidate_preds
        )

        # print(thoughts_query)
        # pdb.set_trace()

        encoded_thoughts_prompt = self.generation_model_tokenizer(
            thoughts_query, return_tensors='pt'
        )['input_ids'].to(self.generation_model.device)

        ## get the tokens of remaining path
        ## add padding and active sequence if necessary
        if len(rest_path) >= 1:
            target_tokens = torch.nn.utils.rnn.pad_sequence(
                [self.generation_model_tokenizer.encode(
                        path, 
                        add_special_tokens=False, 
                        return_tensors='pt'
                    ).squeeze(0) for path in rest_path], 
                batch_first=True,
                padding_value= self.generation_model_tokenizer.eos_token_id
                )
        ## when depth reaches the maximum depth
        elif len(current_path) >= self.max_depth:
            target_tokens = []
        else:
            target_tokens = []
        
        ## get the logprob of the target path
        # mask out the probability based on the current tree
        # print(rest_path, torch.cuda.memory_allocated() / 1e9)
        logp, logeosprob = get_target_tokens_logp(
                    self.generation_model, 
                    encoded_prompt=encoded_thoughts_prompt.repeat(
                                    len(target_tokens), 1
                                    ), 
                    target_tokens=target_tokens,
                    eos_token_id= self.generation_model_tokenizer.eos_token_id, 
                    use_cache=True
                    )
        return logp, logeosprob
    
    def next_path_sampling(
            self,
            logprob: torch.tensor,
            logeosprob : torch.tensor, 
            rest_path : list[str],
            top_k : int,
    )-> Union[list[str], torch.tensor]:
        """Return the k sample and its probability
        Inputs:
            logprobs   : [NUM_PATH, ]
            logeosprob : [1,]   
        """
        logprob = torch.cat([logprob, logeosprob],axis=-1)
        # logprob_normalized = torch.exp(logprob) / torch.exp(logprob).sum(dim=-1, keepdims=True)
        logprob_clone = logprob.clone().detach().cpu().numpy()
        p = np.exp(logprob_clone) / np.exp(logprob_clone).sum(
                                        axis=-1, keepdims=True
                                    )
    
        index = np.random.choice(
            np.arange(0, len(rest_path)+1),   # plus one because of eos  
            top_k,
            p = p,
            replace=False
        )
        sampled_path = [rest_path[i] if i < len(rest_path) else None for i in index]
        # print(logprob, index, top_k, rest_path)
        sampled_logprob = torch.stack([logprob[i] for i in index], dim=0)
        # sampled_logprob = torch.stack([logprob_normalized[i] for i in index], dim=0)

        ## TODO change the sampling to top-k sampling related to logprob  
        return sampled_path, sampled_logprob
    
    def next_path_sampling_uniform(
            self,
            rest_path : list[str],
            top_k : int,
    ):
        """Uniformly sample next path (exploration)"""
        index = np.random.choice(
            np.arange(0, len(rest_path)+1),   # plus one because of eos  
            top_k,
            replace=False
        )

        sampled_path = [rest_path[i] if i < len(rest_path) else None for i in index]
        sampled_logprob = torch.Tensor([np.log(1 / (len(rest_path) + 1))]* top_k).to(self.inference_model.device)

        return sampled_path, sampled_logprob          
    
    def traj_logprob(
            self,
            history,
            curr_state_hist,
            rest_state_hist,
            actions_hist
    ):
        """Return the transition probs of a trajs (length prob = length state hist)"""
        next_state_logprob = torch.zeros(size=(len(curr_state_hist),)).to(
                            self.generation_model.device
                        )
        end_state_logprob = torch.zeros(size=(len(curr_state_hist),)).to(
                            self.generation_model.device
                        )


        for i in range(len(curr_state_hist)):
         
            ns_logprob, es_logprob = self.next_state_logprob(
                history,
                current_state=curr_state_hist[i],
                current_rest_state=rest_state_hist[i],
                actions = actions_hist[i]
            )
            next_state_logprob[i] = ns_logprob
            end_state_logprob[i] = es_logprob

        return next_state_logprob, end_state_logprob


    def next_state_logprob(
            self,
            history,
            current_state : list[list[str]],
            current_rest_state : list[list[str]],
            actions : list[list[str]],
    ):
        """Return the probability of tree transition (off-policy learning)"""
        next_state_logprob = torch.zeros(size=(1,)).to(self.generation_model.device)
        end_state_logprob = torch.zeros(size=(1,)).to(self.generation_model.device)
        
        i = 0
        for j, (cur_path, rest_path) in enumerate(
            zip(current_state, current_rest_state)
            ):
            ## we only consider rest path > 0 and w < max_width 
            ## we will not give action to those has rest path= 0 and w=max_width, so we have to jump it
            """
            action [
                [['8'], []], 
                [[], ['2'], [], ['15']], 
                [[], ['1'], [], ['0'], [], ['2']]
                ]
            """
            if len(rest_path) > 0 and len(cur_path) < self.max_depth:
                leaf_actions = actions[i:i+min(len(rest_path),self.top_k)]
                i += min(len(rest_path),self.top_k)
                # print(action, torch.cuda.memory_allocated() / 1e9) 
                logp, logeosprob = self.next_path_logprob(
                    history, 
                    cur_path,
                    rest_path =  rest_path
                )
                for action in leaf_actions: 
                    if len(action) > 0:
                        try:
                            logp_next_path = logp[rest_path.index(action[0])]
                        except ValueError:
                            print(current_state)
                            print(current_rest_state)
                            print(actions)
                            print("action",action, "rest", rest_path)
                            raise
                        next_state_logprob += logp_next_path
                        end_state_logprob += logeosprob
                    else:
                        ## when action is empty means the branch takes an eos action
                        next_state_logprob += logeosprob
       
        return next_state_logprob, end_state_logprob
        

    def generate_next_state_logprob(
            self,
            history : np.ndarray,
            current_state : list[list[str]]
    ) -> Union[list[list[str]], torch.tensor, torch.tensor, bool]:
        """ Return the next state and its logprob based on the aggregation of current path"""
        next_state_logprob = []
        end_state_logprob = []
        next_state = []
        trunated_dones = []       # when all the path are sampled
        sampled_dones = []        # when dones are sampled
        rest_state = []
        actions = []
        info = {}
        for current_path in current_state:

            rest_path = self.get_rest_path(current_path)
            mask = get_tree_mask(rest_path, current_path, current_state)
            rest_path = [r for r, mask in zip(rest_path, mask.to(torch.bool).tolist()) if not mask]
            rest_state.append(rest_path)
            if len(rest_path) >= 1 and len(current_path) < self.max_depth: 
                logprob, logeosprob = self.next_path_logprob(
                                                history, 
                                                current_path, 
                                                rest_path,
                                            )
                if random.random() > self.epsilon:

                    next_path, next_path_logprob = self.next_path_sampling(
                                logprob, logeosprob, rest_path,
                                top_k=min(self.top_k, len(rest_path))
                            )
                else:
                    ## tempering strategy / epsilon-greedy
                    next_path, next_path_logprob = self.next_path_sampling_uniform(
                        rest_path=rest_path, top_k=min(self.top_k, len(rest_path))
                    )

                ## append the path to the current path
                for path in next_path:
                    sub_action = []
                    cc = current_path.copy()
                    # Exclude None
                    if path:
                        cc.append(path)
                        sub_action.append(path)
                        sampled_dones.append(False)
                    else:
                        sampled_dones.append(True)

                    next_state.append(cc)
                    actions.append(sub_action)
                    if len(rest_path) == 1:
                        trunated_dones.append(True)
                    else:
                        trunated_dones.append(False)
            else:
                next_state.append(current_path)
                trunated_dones.append(True)
                sampled_dones.append(True)
                logeosprob = torch.zeros(size=(1,)).to(self.generation_model.device)
                next_path_logprob = torch.zeros(size=(1,)).to(self.generation_model.device)
            ## store the ending prob for current path
            next_state_logprob.append(next_path_logprob.sum(dim=-1))
            end_state_logprob.append(logeosprob)
        next_state_logprob = torch.stack(next_state_logprob, dim=0)
        end_state_logprob = torch.stack(end_state_logprob, dim=0)
        # print(len(trunated_dones), len(next_state))
        # print("truncate:", trunated_dones)
        # print("sampled dones", sampled_dones)
        
        done = all(trunated_dones) | all(sampled_dones) | all(
            np.array(sampled_dones) | np.array(trunated_dones))

        info['rest_state'] = rest_state
        info['actions'] = actions
        return next_state, next_state_logprob, end_state_logprob, done, info
    
        
    def conditional_prior_logprob(
            self,
            history,
            target_y, 
            state,
    ):

        obs, rationales, sol_prompt = sol_prompt_wrap(
                                            history, 
                                            state,
                                            candidate_preds=self.candidate_preds,
                                            head_preds=self.head_preds,
                                            sol_examples= self.examples
                                        )
        
        # print(sol_prompt)
        # pdb.set_trace()
        thoughts_prior_logprob = self.prior_logprob(
            obs=obs, rationales=rationales, num_sequence=3
        )
        encoded_sol_prompt = self.prior_model_tokenizer(
            sol_prompt, return_tensors='pt'
        )['input_ids'].to(self.prior_model.device)
        # print(sol_prompt)
        ## padding different length of head predicates
        encoded_all_sols = torch.nn.utils.rnn.pad_sequence([
            self.prior_model_tokenizer(
                f"{i}", return_tensors='pt'
            )['input_ids'].to(self.prior_model.device).squeeze(0) for i in self.head_preds 
            ], 
            batch_first=True,
            padding_value= self.prior_model_tokenizer.eos_token_id,
        )
        
        # encoded_complete_answer = torch.cat([
        #     encoded_sol_prompt.repeat(len(encoded_all_sols), 1),
        #     encoded_all_sols
        # ], dim=-1)
        # find the probability of target 
        # if self.learning_prior: 
        #     results = get_target_y(
        #         # model = self.prior_model,
        #         model = self.prior_model,
        #         encoded_prompt=encoded_complete_answer,
        #         skip_first= encoded_sol_prompt.size(-1),
        #         eos_token_id= self.prior_model_tokenizer.eos_token_id
        #     )
        #     y_logprob = results[self.head_preds_idx.index(target_y)]
        #     print("target prob", y_logprob,self.candidate_preds[target_y])
    
        # else:
        #     y_logprob = 0
        return thoughts_prior_logprob
    
        # return thoughts_prior_logprob

    @torch.inference_mode()
    def logreward(
            self,
            history : np.ndarray,
            target_y : int,
            state: list[list[str]],
            logic_model : nn.Module
    ):
        """Return the log-reward given history x and thoughts 
        """
        info = {}   
        ## whether to update p(y | R, X) or not
        tokenizer = self.inference_model_tokenizer
        model = self.inference_model

        obs, rationales, sol_prompt = sol_prompt_wrap(
                                            history, 
                                            state,
                                            candidate_preds=self.candidate_preds,
                                            head_preds=self.head_preds,
                                            sol_examples= self.examples
                                        )
        ## TODO check whether the node is leaf node or not
        print(state)
        print(rationales)
        # print(sol_prompt)
        # pdb.set_trace()
        encoded_sol_prompt = tokenizer(
            sol_prompt, return_tensors='pt'
        )['input_ids'].to(model.device)
        # print(sol_prompt)
        # print(state)
        ## padding different length of head predicates
        encoded_all_sols = torch.nn.utils.rnn.pad_sequence([
            tokenizer(
                f"{i}", return_tensors='pt'
            )['input_ids'].to(model.device).squeeze(0) for i in self.head_preds 
            ], 
            batch_first=True,
            padding_value= tokenizer.eos_token_id,
        )
        
        encoded_complete_answer = torch.cat([
            encoded_sol_prompt.repeat(len(encoded_all_sols), 1),
            encoded_all_sols
        ], dim=-1)
        ## find the probability of target 
        results = get_target_y(
            model = model,
            # model = self.prior_model,
            encoded_prompt=encoded_complete_answer,
            skip_first= encoded_sol_prompt.size(-1),
            eos_token_id= tokenizer.eos_token_id
        )
        # pdb.set_trace()
        # print(sol_prompt)
        print("y_prob:", results)
        print("target", self.candidate_preds[target_y])
        print("predict", self.head_preds[torch.argmax(results)])
        # results = F.softmax(torch.exp(results), dim = -1)
        
        # p(R)
        thoughts_prior_logprob = self.prior_logprob(obs=obs, rationales=rationales)
        pred_y = self.head_preds_idx[torch.argmax(results)]
        # p (y | R, X)
        # target_y_logprob = torch.exp(
        #     results[self.head_preds_idx.index(target_y)]
        # )
        # print(target_y, self.head_preds_idx )
        # print(pred_y, target_y)
        target_y_logprob = results[self.head_preds_idx.index(target_y)]

        
        rank = calc_rank(
            label = [self.head_preds_idx.index(target_y)],
            pred  = results.unsqueeze(dim=0).cpu().numpy()
        ).item()
       
        # rank =  torch.nonzero(
        #             torch.sort(
        #                 results, 
        #                 descending=True
        #             ).values == target_y_logprob
        #             ).item() + 1
        acc = pred_y == target_y
        # print(target_y, self.head_preds_idx)
        ## p (x | R,)
        llh = logic_model.log_likelihood(
                t = max(history[:,0]),
                X = np.expand_dims(history, axis=0), ## batch the data
                predicates = [p for p in self.head_preds_idx],
                rules =  state_to_rules(
                    self._strstate_to_intstate(state),
                    max_rule_length= self.max_depth
                )
            )
        info['acc'] = acc
        info['rank'] = rank
        # p (y | R, X) p(R) p(x | R)
        # print(f"y {target_y}, y_logp {target_y_logprob}, llh {llh}, prior {thoughts_prior_logprob.item()}")
        log_reward = target_y_logprob + llh + thoughts_prior_logprob.item()
        return  log_reward, llh, info
    
    def _strstate_to_intstate(
            self,
            state,
    ):
        """convert the state representation to int """
        state_idx_rep = []
        for path in state:
            state_idx_rep.append(
                [self.candidate_preds.index(preds) for preds in path]
            )
        return state_idx_rep


    def prior_logprob(
            self,
            obs : str,
            rationales : str, 
            num_sequence : int = 1,
    ):
        """A prior for thoughts"""
        prior_thoughts = obs + prompt_thought_prior
        encoded_prior_prompt = self.prior_model_tokenizer(
            prior_thoughts, return_tensors='pt'
        )['input_ids'].to(self.prior_model.device)

        encoded_thoughts = self.prior_model_tokenizer(
            rationales, return_tensors='pt'
        )['input_ids'].to(self.prior_model.device)
     
        encoded_complete_answer = torch.cat(
            [encoded_prior_prompt, encoded_thoughts], 
            dim=-1
        )

        ## find the probability of target 
        results = get_target_y(
            model = self.prior_model,
            encoded_prompt=encoded_complete_answer.repeat(num_sequence,1),
            skip_first= encoded_prior_prompt.size(-1),
            eos_token_id= self.prior_model_tokenizer.eos_token_id
        )
      
        return results

    # @staticmethod
    # def propose_prompt_wrap(
    #         x: np.ndarray, 
    #         y_list: list[str],
    #         rest_event : list[str],
    #         candidate_preds : list[str]
    # ) -> str:
    #     """
    #     Inputs:
    #     x  : event history in a numerical form stored in an array
    #             i.e., [[0.23, 0], [0.4, 1]]
    #     y_list : a ordered list strings stores current path
    #         the left most string is the intial target,
    #         the right most string is the current target 
    #         the sub-target is separated by empty string, 
    #             i.e., ['4', '3', '2']
    #     rest_events : a ordered list of strings store rest path
    #             i.e., ['0', '1']
    #     candidate_preds : a list of all available preds 
    #     Returns:
    #         A propose prompt for GPT input
    #     """

    #     rest_history = x.copy()
    #     head_event = y_list[-1]
    #     assert head_event in candidate_preds, f"Head event {head_event} not found"

    #     for i in range(0, len(y_list)-1):
    #         ## return rest history of previous target
    #         rest_history = rest_history[rest_history[:,1]!=int(
    #             candidate_preds.index(y_list[i])
    #         )]

    #     rest_observation = prompt_observation.format(
    #         total_events = ','.join(rest_event + [head_event]),
    #         events_history = events2text(rest_history, candidate_preds)
    #     )
        
    #     prompt_thoughts_inst = prompt_thoughts.format(
    #         head_event = head_event,
    #         # possible_events = ','.join(rest_event)
    #     )
    #     ## form prompt
    #     prompt_thoughts_inst = rest_observation + prompt_thoughts_inst
    #     return prompt_thoughts_inst
    
    # @staticmethod
    # def sol_prompt_wrap(
    #         x : np.ndarray,
    #         rationales: list[list[str]],
    #         candidate_preds : list[str],
    #         head_preds : list[str],
    # ):
    #     """ Return the solution prompt given accumated rationales and historical data X"""
    #     all_observation = prompt_observation.format(
    #         total_events = ','.join(candidate_preds),
    #         events_history = events2text(x, candidate_preds),
    #     )

    #     rule_text_list = []
    #     for rule in rationales:
    #         rule_text = rule2text(rule)
    #         if len(rule_text) > 3:
    #             rule_text_list.append(rule_text)

    #     rationales_text = "".join([f"{i+1}. {text} \n" for i, text in enumerate(rule_text_list) ])

    #     prompt_sol_inst = prompt_sol.format(
    #         rationales = rationales_text,
    #         time = round(np.max(x,axis=0)[0], 2),
    #         possible_events = ','.join(head_preds)
    #     )

    #     return all_observation, rationales_text, all_observation + prompt_sol_inst
    
    # @staticmethod
    # def state_to_rules(current_state):
    #     """To integer representation in events"""
    #     rules = []

    #     for path in current_state:
    #         for s, e in zip(path[::-1][:-1], path[::-1][1:]):
    #             # assume the predicates are integer
    #             rules.append([int(s), int(e)])

    #     return rules