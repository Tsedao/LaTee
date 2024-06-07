import os
import json
# os.environ["TRANSFORMERS_CACHE"] = '/data/home/zitaos/llm/huggingface/hub'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import math
import wandb
import random

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from typing import Union
from functools import partial


from model import EventGPTGFN, db_loss, subtb_loss
from logic_model import LogicModel, LogicModelTrainer
from train_datasets import SyntheticEventdata

import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.data_utils import (
    array_2_list,
    collate_fn,
)

from replay_buffer import ReplayBuffer

from utils.train_utils import get_lora_model, load_model
from utils.llm_utils import state_to_rules, get_path_logprob

def init_before_training(seed=3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description='EventGPTGFN')

    # Add arguments
    parser.add_argument('--gpu', type=str, help='gpu id')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_depth',type=int, default=4)
    parser.add_argument('--max_width',type=int, default=5)
    parser.add_argument('--llm_lr', type=float, default=5e-4)
    parser.add_argument('--logic_model_lr', type=float, default=0.01)
    parser.add_argument('--lm_update_steps', type=int, default=1)
    parser.add_argument('--alternate_every', type=int, default=10)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--llm_size', type=str, default='medium')
    parser.add_argument('--inf_llm_size', type=str, default='mistral-7b')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--no_semantic', action='store_true')
    parser.add_argument('--learn_prior', action='store_true')
    parser.add_argument('--restore_gt_rules', action='store_true')
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--n_shots', type=int, default=3)
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    
    
    # Parse the arguments
    args = parser.parse_args()


    ## gfn-tot args
    max_depth = args.max_depth
    max_width = args.max_width
    topk = args.topk

    ## train args
    seed = args.seed 
    gpu = args.gpu
    lr = args.llm_lr
    prior_lr = args.llm_lr
    logic_model_lr = args.logic_model_lr
    logic_model_update_steps = args.lm_update_steps
    warmup = args.warmup
    warmup_steps = max(2 * args.alternate_every,20)
    epsilon = args.epsilon

    # total_steps = 1000
    total_epoches = args.epoches
    bs = args.bs
    learn_prior = args.learn_prior
    n_shots = args.n_shots 

    loss_fun_dict ={
        "subtb_loss" : subtb_loss,
        "db_loss" : db_loss
    }
    loss_fun_name = "subtb_loss"
    loss_fun = loss_fun_dict[loss_fun_name]

    ## eval args
    eval_steps = 5


    ## data args
    data_path = None
    # horizon = 5
    # data_name = 'synthetic'
    # data_name = 'so'
    data_name = args.dataset
    # data_name = 'mimic2'


    init_before_training(seed)
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    # model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    # model_name = 'nlpcloud/instruct-gpt-j-fp16'

    # model_name = 'stabilityai/stablelm-zephyr-3b'
    if args.llm_size == 'large':
        model_name = 'facebook/opt-6.7b'
    elif args.llm_size == 'medium':
        model_name = 'facebook/opt-1.3b'
    elif args.llm_size =='zephyr-3b':
        model_name = 'stabilityai/stablelm-zephyr-3b'
    elif args.llm_size == 'mistral-7b':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    elif args.llm_size == 'small':
        model_name = 'facebook/opt-350m'
    else:
        raise NotImplementedError
    
    if args.inf_llm_size == 'mistral-7b':
        inf_model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    elif args.inf_llm_size == 'medium':
        inf_model_name = 'facebook/opt-1.3b'
    elif args.inf_llm_size == 'zephyr-3b':
        inf_model_name = 'stabilityai/stablelm-zephyr-3b'
    else:
        raise NotImplementedError
    
    # elif args.llm_size == 'large':
    #     model_name = 'stabilityai/stablelm-zephyr-3b'
    # model_name = 'facebook/opt-125m'
    # model_name = 'Neko-Institute-of-Science/LLaMA-7B-HF'
    # model_name = 'TheBloke/Llama-2-7b-Chat-GPTQ'

    prior_model_name = 'facebook/opt-350m'
    # cache_dir = '/data/home/zitaos/llm/huggingface/hub'
    cache_dir = '/home/zitao.song/.cache/huggingface/hub'


    model, tokenizer = load_model(
                model_name=model_name,
                cache_dir= cache_dir, 
                device= device,
            )
    # model.to(device)
    
    ## init gfn generator/sampler with samller rank 
    generation_model = get_lora_model(
        model=model,
        r = 512,
        # r = 128,
        lora_alpha = 512,
        lora_dropout = 0
    )
    prior_model, prior_tokenizer = load_model(
                model_name=prior_model_name ,
                cache_dir= cache_dir,
                device = device,
            )
    ## Comment out .to(device) in 4-bit or 8-bit learning
    # prior_model.to(device)

    ### init prior inference model with smaller rank
    prior_generation_model = get_lora_model(
        model=prior_model,
        r = 512,
        lora_alpha = 512,
        lora_dropout = 0
    )

    inference_model, inference_model_tokenizer = load_model(
        model_name=inf_model_name,
        cache_dir = cache_dir,
        device = device
    )

    with open(f'data/{data_name}/{data_name}_x.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(f'data/{data_name}/{data_name}_y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open(f'data/{data_name}/{data_name}_configs.json', 'r') as f:
        data_configs = json.load(f)

    head_predicates = data_configs['head_preds']
    body_predicates = data_configs['body_preds']
    head_predicates_name = data_configs.get(
        'head_preds_name',
        [str(i) for i in head_predicates]
    )
    body_predicates_name = data_configs.get(
        'body_preds_name',
        [str(i) for i in body_predicates]
    )
    
    if args.no_semantic:
        head_predicates_name = [str(i) for i in head_predicates]
        body_predicates_name = [str(i) for i in body_predicates]
    
    # horizon = int(data_configs.get('horizon', 5)) + 1
    

    X, y = X[:min(len(X),400)], y[:min(len(y),400)]  # crop the data to save time
    # train_num = int(0.8 * len(X))
    # test_num = len(X) - train_num // 2 
    # train_X, test_X = X[:train_num], X[train_num:]
    # train_y, test_y = y[:train_num], y[train_num:]
    # eval_X, test_X = test_X[:test_num], test_X[test_num:]
    # eval_y, test_y = test_y[:test_num], test_y[test_num:]
    train_X, test_X, train_y, test_y = train_test_split(
                    X  ,y, test_size=0.2, random_state=seed,
                )
    eval_X, test_X, eval_y, test_y = train_test_split(
                test_X, test_y, test_size=0.5, random_state=seed,
                )
    
    logic_model = LogicModel( 
            head_predicates=head_predicates,
            body_predicates=body_predicates,
            rule_max_length=max_depth,
        )
    if args.restore_gt_rules:
        logic_model.restore_gt_rules(
            gt_rules = data_configs['rules'],
            gt_rules_weights = data_configs['rule_weights'],
            gt_body_mu = data_configs['body_mu'],
            gt_head_mu= data_configs['head_mu'],
        )

    logic_model_trainer  = LogicModelTrainer(
                        logic_model,
                        train_params= {
                            "lr":logic_model_lr,
                            "T_max": args.alternate_every,
                        },
                        # data_params= {"horizon": horizon}
                        )
    gfn = EventGPTGFN(
       generation_model=generation_model, 
       generation_model_tokenizer=tokenizer,
       inference_model = inference_model,
       inference_model_tokenizer = inference_model_tokenizer,
       prior_model= prior_generation_model,
       prior_model_tokenizer= prior_tokenizer,
       knowledge_model=logic_model,
       head_preds={i : name for i, name in zip(head_predicates, head_predicates_name)},
       body_preds={i : name for i, name in zip(body_predicates, body_predicates_name)},
       max_depth=max_depth,
       max_width=max_width,
       topk = topk,
       epsilon = epsilon,
       learning_prior=learn_prior,
       example_nums= n_shots
    )

    # learning rate schedule
    opt = torch.optim.AdamW(
        [{'params': generation_model.parameters(), 'lr': lr}], 
        betas=(0.9, 0.99)
        )
    
    prior_opt = torch.optim.AdamW(
        [{'params': prior_generation_model.parameters(), 'lr': prior_lr}], 
        betas=(0.9, 0.99)
    )

    replay_buffer = ReplayBuffer(capacity=1000)

    total_steps = total_epoches * (len(train_X) // bs + 1) // 2
    def get_lr_mult_at_step(step):
        if step <= warmup_steps:
            return min(step/warmup_steps, 1.)
        return max(0.995*((total_steps - step) / (total_steps - warmup_steps)), 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_mult_at_step)
    sched_prior = torch.optim.lr_scheduler.LambdaLR(prior_opt, get_lr_mult_at_step)

    now = datetime.now()

    current_time = now.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_name}_{data_name}"
    experiment_name += f"_lr_{lr}_bs_{bs}_warmup{warmup}_w_{max_width}"
    experiment_name += f"_d_{max_depth}_{loss_fun_name}" 
    experiment_name += current_time
    # print(f"Saving experiment results to {experiment_name}")
    # writer = SummaryWriter(f"./experiments/{experiment_name}.log")

    wandb.login(
        key = ''
    )
    config = vars(args)
    config['loss_fun'] = loss_fun_name
    run = wandb.init(
        # Set the project where this run will be logged
        project="backward_reasoning_llm",
        notes = experiment_name,
        tags= [
            args.dataset, 
            f"inf: {args.llm_size}",
            f"est: {args.inf_llm_size}", 
            "no_semantic" if args.no_semantic else "semantic",
            "learn_prior" if args.learn_prior else "not_learn_prior"
        ],
        # Track hyperparameters and run metadata
        config=config,
    )

    wandb.define_metric("llh_loss/train", summary="min")
    wandb.define_metric("loss/train", summary="min")
    wandb.define_metric("acc/train", summary="mean")
    wandb.define_metric("rewards/train", summary="max")
    wandb.define_metric("acc/eval", summary="max")
    wandb.define_metric("acc/test", summary="max")
    wandb.define_metric("llh/eval", summary="max")

    s = 0
    for e in range(total_epoches):

        
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        batches = [indices[i:min(i + bs, len(indices))] for i in range(0, len(indices), bs)]
        
        for indice in batches:

            step_loss = 0
            llh_step_loss = 0
            logrewards_avg = 0
            acc = 0
            rank = 0
            now = time.time()

            batch_x = [train_X[i] for i in indice]
            batch_y = [train_y[i] for i in indice]

            # Alternate between E and M steps
            s += 1
            if s % (2*args.alternate_every) < (args.alternate_every):
                stage = 'dec'
            else:
                stage = 'gfn'
            
            if args.restore_gt_rules:
                stage = 'gfn' ## only E-steps

            if stage == 'gfn':
                opt.zero_grad()
                logic_model.eval()

                ## E-step
                for x_sample, y_sample in zip(batch_x,batch_y):
                    # choose a behavior policy
                    b_policy_choice = random.randint(0, 6)
                    # b_policy_choice = -1
                    if len(replay_buffer) > 0 and b_policy_choice in [0, 1] and args.explore:
                        # use samples from the replay buffer
                        experience = replay_buffer.sample()[0]
                        curr_state_hist, rest_state_hist, action_hist = experience.trajectory
                        logrewards = experience.reward
                        x_sample, y_sample = experience.query
                        logpf, logeosprobs = gfn.traj_logprob(
                            history=x_sample,
                            curr_state_hist=curr_state_hist,
                            rest_state_hist=rest_state_hist,
                            actions_hist = action_hist
                        )

                    else:
                        # use the action policy without tempering
                        (current_state, logrewards, logeosprobs, 
                            logpf, llh, preds_info, hist_info) = gfn.act(x_sample,y_sample)
                        replay_buffer.add(
                            query= (x_sample, y_sample),
                            trajectory= (
                                hist_info['curr_state_hist'],
                                hist_info['rest_state_hist'],
                                hist_info['actions_hist'],
                            ),
                            reward= logrewards,
                        )


                    loss = loss_fun(
                        logrewards.to(generation_model.device), 
                        logpf, 
                        logeosprobs
                    ).mean()
                    
                    loss.backward()
                    
                    step_loss += loss.item()
                    del loss ## release memory
                    ## retrieve the last prediction done
                    last_info = preds_info[-1]
                    acc_, rank_ = last_info['acc'], last_info['rank']
                    acc += int(acc_)
                    rank += rank_
                    logrewards_avg += logrewards.mean()

                # print("rule_weights", logic_model.rule_weights)
                opt.step()                # E-step
                sched.step() if warmup else None
                logrewards_avg = logrewards_avg.to(torch.float32)
                wandb.log({
                    "loss/train": step_loss/ bs,
                    "lr_gen_model/train" : opt.param_groups[0]['lr'],
                    "rewards/train": logrewards_avg / bs,
                    "acc/train": acc / bs,
                    "rank/train" : rank / bs,
                })

                print(
                    f"E-step| "
                    f"Time elapsed : {time.time()- now :.2f}| "
                    f"Step :{e}, loss : {step_loss/ bs:.3f}| " 
                    f"Prediction acc : {acc / bs:3f}| "
                    f"Mean rank : {rank / bs :.3f}| "
                    f"rewards : {logrewards_avg / bs:.3f}| "
                    )
            elif stage == 'dec':
            # M-step for learnable prior
                logic_model.train()
                prior_loss_sum = 0
                prior_opt.zero_grad()
                logic_model_trainer.opt.zero_grad()
                for x_sample, y_sample in zip(batch_x,batch_y):
                    with torch.inference_mode():
                        (current_state, logrewards, logeosprobs, 
                        logpf, llh, preds_info_dec, hist_info) = gfn.act(x_sample,y_sample)
                    state_history = hist_info['next_state_hist']
                    # print(logpf)
                    # print(logeosprobs)
                    # print(hist_info['curr_state_hist'])
                    # print(hist_info['actions_hist'])
                    # print(hist_info['rest_state_hist'])
                    # next_state_logprob, end_state_logprob = gfn.traj_logprob(
                    #     history=x_sample,
                    #     curr_state_hist=hist_info['curr_state_hist'],
                    #     rest_state_hist=hist_info['rest_state_hist'],
                    #     actions_hist = hist_info['actions_hist']
                    # )
                    # print(next_state_logprob)
                    # print(end_state_logprob)

                    if learn_prior:
                        for state in state_history:
                            prior_logprob = gfn.conditional_prior_logprob(
                                history  = x_sample,
                                target_y = y_sample,
                                state    = state,
                            )
                            # prior_loss = - (prior_logprob + 10 * y_logprob).mean()
                            prior_loss = - prior_logprob.mean()
                            prior_loss.backward()
                            prior_loss_sum += prior_loss.item()
                            del prior_loss ## release memory

                    # M-step for knowledge model
                    llh_loss = logic_model_trainer.train(
                        batch_x=[x_sample],
                        steps = logic_model_update_steps,
                        rules= state_to_rules(
                            gfn._strstate_to_intstate(
                                current_state
                            ))
                    )
                    llh_step_loss += llh_loss

                prior_opt.step()          # M-step
                logic_model_trainer.opt.step()
                print(logic_model_trainer.model.rule_weights.sum(), logic_model_trainer.model.rule_weights.size())
                logic_model_trainer.lr_sched_step()
            

                # writer.add_scalar("llh_loss/train", llh_step_loss/ bs, e)     
                # writer.add_scalar("loss/train", step_loss/ bs, e)
                # writer.add_scalar("acc/train", acc / bs, e)
                # writer.add_scalar("rewards/train", logrewards_avg / bs, e)

                wandb.log({
                    "llh_loss/train": llh_step_loss/ bs,
                    "lr_logic_model/train" : logic_model_trainer.opt.param_groups[0]['lr'],
                })


                print(
                    f"M-step| "
                    f"Time elapsed : {time.time()- now :.2f}| "
                    f"llh loss : {llh_step_loss/ bs:.3f}| "
                    f"prior loss: {prior_loss_sum / bs:.3f}| " 
                    )
            # torch.cuda.empty_cache()

        ### evaluating after one epoch
        eval_time = time.time()
        # if e % eval_steps == 0:
            
        llh_sum = 0
        acc_sum = 0
        rank_sum = 0
        entropy = 0
        for x_eval, y_eval in zip(eval_X, eval_y):
            llh, eval_pred_info = gfn.eval(x_eval, y_eval)
            llh_sum += llh
            acc_sum += eval_pred_info['acc']
            rank_sum += eval_pred_info['rank']
            entropy += eval_pred_info['entropy_hist'][-1].item()

        # writer.add_scalar("llh/eval", llh_sum / len(test_X), e)
        # writer.add_scalar("acc/eval", acc_sum / len(test_X), e)

        wandb.log({
            "llh/eval": llh_sum / len(eval_X),
            "acc/eval": acc_sum / len(eval_X),
            "rank/eval":rank_sum / len(eval_X),
            "entropy/eval" : entropy / len(eval_X)
        })

        print(
        f"Eval: "
        f"Time elapsed : {time.time()- eval_time :.2f}| "
        f"Prediction acc : {acc_sum / len(eval_X):.3f}| "
        f"Entropy : {entropy / len(eval_X):.3f}| "
        f"llh : {llh_sum / len(eval_X):.3f}| "
        )
        # torch.cuda.empty_cache()
        logic_model_weights_path = f"models/logic_model/{data_name}"
        os.makedirs(logic_model_weights_path,exist_ok=True)
        logic_model_trainer.save_rule_weights(logic_model_weights_path,name=f'e{e}')
    ## evaluation on test dataset
    llh_sum = 0
    acc_sum = 0
    rank_sum = 0
    for x_eval, y_eval in zip(test_X, test_y):
        llh, test_pred_info = gfn.eval(x_eval, y_eval)
        llh_sum += llh
        acc_sum += test_pred_info['acc']
        rank_sum += test_pred_info['rank']

    wandb.log({
        "llh/test": llh_sum / len(test_X),
        "acc/test": acc_sum / len(test_X),
        "rank/test": rank_sum / len(test_X)
    })
        
    # writer.flush()

    if args.save:
        logic_model_weights_path = f"models/logic_model/{data_name}"
        print(f"saving logic model to {logic_model_weights_path}")
        os.makedirs(logic_model_weights_path,exist_ok=True)
        logic_model_trainer.save_weights(
            os.path.join(
                logic_model_weights_path,
                f"logic_model.pt",
            )
        )
        logic_model_trainer.save_rule_weights(logic_model_weights_path)

        generation_model_weights_path = f"models/generation_model/{data_name}"
        print(f"saving inference model to {generation_model_weights_path}")
        os.makedirs(generation_model_weights_path, exist_ok=True)
        generation_model.save_pretrained(os.path.join(
            generation_model_weights_path,
            f"generation_model.pt"
        ))