import torch
import numpy as np

from collections import defaultdict

from typing import Union

from utils.prompt_template import (
    prompt_thoughts,
    prompt_sol,
    prompt_observation,
    prompt_sols_with_rules,
)

import utils.prompt_examples as sol_prompt_examples
import utils.path_prompt_examples as path_prompt_examples

import pdb

def categorize_paths(paths):
    tree = {}
    terminating_paths = []
    non_terminating_paths = []

    # Build the tree from paths
    for path in paths:
        current = tree
        for node in path:
            if node not in current:
                current[node] = {}
            current = current[node]

    # Function to determine if a node is a terminating node
    def is_terminating(node):
        return len(node) == 0

    # Find paths leading to terminating and non-terminating nodes
    def find_paths(current, path):
        # Check if current node is terminating
        if is_terminating(current):
            terminating_paths.append(path)
        else:
            non_terminating_paths.append(path)
        
        for node in current:
            find_paths(current[node], path + [node])

    # Start from the root node(s), should be handled dynamically
    for root in tree:
        find_paths(tree[root], [root])

    return terminating_paths, non_terminating_paths

def events2text(events, preds_name):
    """Convert numerical events to text"""
    events_dict = defaultdict(lambda : [])
    ## find the historical event for each type and store in dict
    for sample in events:
        events_dict[sample[1]].append(sample[0])
    text_out = ""
    for i, (event, recs) in enumerate(events_dict.items()):

        activate_time = [str(round(t, 2)) for t in recs ]

      
        act_comb = f"{i+1}. Event {preds_name[int(event)]} happens at time {', '.join(activate_time)}"

        text_tmp = f"{act_comb} \n"
        text_out += text_tmp
    return text_out

def rule2text(y_list : list[str]) -> str:
    """Convert a list a rules to text"""

    ## from backward reasoning to forward reasoning
    rule_head = f"Event {y_list[0]} <-- "
    temporal_logic =  " and ".join([f"(Time of Event {a} after Time of Event {b})" for a, b in zip(
                                    y_list[:-1],
                                    y_list[1:],
                            )])
    unitary_logic = " and ".join([f"(Event {a})" for a in y_list[1:]])

    return rule_head + unitary_logic + " and " + temporal_logic

def append_sol_and_remove_eos(text : torch.Tensor, result : torch.Tensor, eos_token_id : int, pad_token_id : int):
    # remove anything after the first eos token and append the result
    # if there is no eos token, append the result
    # text is a torch tensor with the first dimension being the batch
    # result is a torch tensor with the first dimension being the batch
    # this is a vectorized implementation
    # returns a torch tensor with the first dimension being the batch
    # and the second dimension being the length of the sequence
    new_text = []
    for t, r in zip(text, result[:text.size(0)]):
        if eos_token_id not in t:
            new_text.append(t if r is None else torch.cat([t, r]))
            continue
        # find the first eos token
        t[(t == eos_token_id).cumsum(dim=-1) >= 1] = eos_token_id
        eos_ind = ((t == eos_token_id).cumsum(dim=-1) == 1).nonzero()[0]
        # remove the eos tokens from the result and shift the result to the left
        if r is not None:
            new_text.append(torch.cat([t[:eos_ind], r]))
        else:
            new_text.append(t[:eos_ind])
    return torch.nn.utils.rnn.pad_sequence(new_text, batch_first=True, padding_value=pad_token_id)



def get_target_y(
        model,
        encoded_prompt : torch.tensor,
        skip_first, 
        eos_token_id 
):
    """return the target y's logpf in LLM"""
    out = model(
            encoded_prompt, 
            attention_mask=encoded_prompt!=eos_token_id
        )
    ## catch the exception of AutoCausalLM with head
    if isinstance(out, tuple):
        logits = out[0][:,:-1,:]
    else:
        logits = out.logits[:,:-1,:]
    
    # get rid of the first few tokens
    # pdb.set_trace()
    logits = logits[:, skip_first-1:]
    logprob = logits.log_softmax(-1)
    token_ids = encoded_prompt[:, skip_first:].unsqueeze(-1)
    logPF = logprob.gather(-1, token_ids).squeeze(-1)
    # change the log probability of eos to 0 (we padding sequence using eos)
    logPF[encoded_prompt[:, skip_first:] == eos_token_id] = 0.
    res = logPF.sum(dim=-1)

    return res

def get_target_tokens_logp(
        model, 
        encoded_prompt : torch.tensor, 
        target_tokens : torch.tensor,
        eos_token_id : int,
        use_cache : bool = True,
    ) -> Union[torch.tensor, torch.tensor]:
    """Return the logprob and logprobeos of a list target tokens """
    # print(torch.cuda.memory_summary())  # Initial memory usage

    state = encoded_prompt.clone()
    target_logp = encoded_prompt.new_zeros(encoded_prompt.size(0)).float()
    logpf = encoded_prompt.new_zeros(target_tokens.size(0), target_tokens.size(1)).float()
    active_target_tokens = (target_tokens != eos_token_id).to(encoded_prompt).float()
    logeosprobs = encoded_prompt.new_zeros(1).float()
    ## when one path reaches end
    if len(target_tokens) == 0:
        return target_logp, logeosprobs
    ## when the path can extend, target tokens not empty
    # print(state.shape)
    if use_cache:
        # print(model(state[:, :-1]))
        past_key_values = model(state[:, :-1]).past_key_values
        # print(torch.cuda.memory_summary())  # Initial memory usage
    for i in range(target_tokens.size(1)):
        if use_cache:
            output = model(state[:, -1:], past_key_values=past_key_values)
        else:
            output = model(state)

        if use_cache:
            past_key_values = output.past_key_values

        ## catch the exception of AutoCausalLMWithHead
        if isinstance(output, tuple):
            logprob = output[0][:,-1,:].log_softmax(dim=-1)
        else:
            logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        # logprob = output['logits'][:, -1, :].log_softmax(dim=-1)
        token_ids_pt =  target_tokens[:,i:(i+1)].to(encoded_prompt)
        # print(tokenizer.batch_decode(token_ids_pt))
        # print(torch.cuda.memory_summary())  # Initial memory usage
        # target_logp += logprob.gather(-1, token_ids_pt).squeeze(-1)
        logpf[:,i] = logprob.gather(-1, token_ids_pt).squeeze(-1)
        state = torch.cat([state, token_ids_pt], dim=-1)

        if i == 0:
            ## record the EOS probability at the beginning, only get the first prompt
            logeosprobs = logprob[:,  eos_token_id]

    ## mask the logpf when target tokens contains eos
    # print(active_target_tokens)
    logpf = logpf * active_target_tokens
    target_logp = logpf.sum(dim=-1)   # sum over the target token dimensions

    return target_logp, torch.mean(logeosprobs, dim=-1, keepdim=True)


def propose_prompt_wrap(
        x: np.ndarray, 
        y_list: list[str],
        rest_event : list[str],
        candidate_preds : list[str],
        examples : str = "",
) -> str:
    """
    Inputs:
    x  : event history in a numerical form stored in an array
            i.e., [[0.23, 0], [0.4, 1]]
    y_list : a ordered list strings stores current path
        the left most string is the intial target,
        the right most string is the current target 
        the sub-target is separated by empty string, 
            i.e., ['4', '3', '2']
    rest_events : a ordered list of strings store rest path
            i.e., ['0', '1']
    candidate_preds : a list of all available preds 
    examples : sentence demonstrations 
    Returns:
        A propose prompt for GPT input
    """

    rest_history = x.copy()
    head_event = y_list[-1]
    assert head_event in candidate_preds, f"Head event {head_event} not found"

    for i in range(0, len(y_list)-1):
        ## return rest history of previous target
        rest_history = rest_history[rest_history[:,1]!=int(
            candidate_preds.index(y_list[i])
        )]

    rest_observation = prompt_observation.format(
        task_instruction = """I want you to do the reasoning over social events.\nYou should use the reasoning criterion that for one event to cause another, its activation must occur before the other,\nYou should also consider the sequence and proximity of activations to determine the "most likely" cause.""",
        examples = examples,
        total_events = ','.join(rest_event + [head_event]),
        events_history = events2text(rest_history, candidate_preds)
    )
    
    prompt_thoughts_inst = prompt_thoughts.format(
        head_event = head_event,
        possible_events = ','.join(rest_event)
    )
    ## form prompt
    prompt_thoughts_inst = rest_observation + prompt_thoughts_inst
    return prompt_thoughts_inst


def sol_prompt_wrap(
        x : np.ndarray,
        rationales: list[list[str]],
        candidate_preds : list[str],
        head_preds : list[str],
        sol_examples : str = ""
):
    """ Return the solution prompt given accumated rationales and historical data X"""
    task_instruction = 'I want you to perform inference over social events.'
    all_observation = prompt_observation.format(
        task_instruction = task_instruction,
        total_events = ','.join(candidate_preds),
        events_history = events2text(x, candidate_preds),
        examples = ""
    )
    rationales, _ = categorize_paths(rationales)
    rule_text_list = []
    for rule in rationales:
        rule_text = rule2text(rule)
        if len(rule_text) > 3:
            rule_text_list.append(rule_text)

    rationales_text = "".join([f"{i+1}. {text} \n" for i, text in enumerate(rule_text_list) ])

    
    # prompt_sol_inst = prompt_sol.format(
    #     rationales = rationales_text,
    #     time = round(np.max(x,axis=0)[0], 2),
    #     possible_events = ','.join(np.random.permutation(head_preds))  ## random shuffle the order the head predicates
    # )

    sol_out_prompt = prompt_sols_with_rules.format(
        examples = sol_examples,
        rationales = rationales_text,
        total_events = ','.join(candidate_preds),
        events_history = events2text(x, candidate_preds),
        time = round(np.max(x,axis=0)[0], 2),
        ## random shuffle the order the head predicates
        possible_events = ','.join(np.random.permutation(head_preds)) 
    )

    return all_observation, rationales_text, sol_out_prompt


def state_to_rules(
        current_state,
        max_rule_length = 3,
    ):
    """To integer representation in events"""
    rules = []

    # for path in current_state:
    #     for s, e in zip(path[::-1][:-1], path[::-1][1:]):
    #         # assume the predicates are integer
    #         rules.append([int(s), int(e)])
    
    for path in current_state:
        path = path[:max_rule_length]
        ## reverse the order
        rules.append([int(e) for e in path[::-1]])
        

    return rules

@torch.inference_mode()
def get_y(
        model,
        tokenizer,
        history : np.ndarray,
        rules : list[list[str]],
        candidate_preds: list[str],
        head_preds : list[str],
        device : str
):
    """return the logprob of target y given generated rules"""
    obs, rationales, sol_prompt = sol_prompt_wrap(
                                        history, 
                                        rules,
                                        candidate_preds=candidate_preds,
                                        head_preds=head_preds
                                    )

    encoded_sol_prompt = tokenizer(
        sol_prompt, return_tensors='pt'
    )['input_ids'].to(device)

    encoded_all_sols = torch.nn.utils.rnn.pad_sequence([
        tokenizer(
            f"{i}", return_tensors='pt'
        )['input_ids'].squeeze(0).to(device)  for i in head_preds 
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
    return results

@torch.inference_mode()
def get_path_logprob(
        model,
        tokenizer,
        history : np.ndarray,
        path : list[str],
        predicates : list[str],
        device : str,
        use_cache : bool=True
    )->torch.tensor:
    """get the marginal logprob of one path"""
    logp_sum = torch.zeros(size=(1,)).to(device)
    for i in range(1, len(path)):
        curr_path = path[:i]
        rest_event = [p for p in predicates if p not in curr_path]

        thoughts_query = propose_prompt_wrap(
            x=history,
            y_list=curr_path,
            rest_event=rest_event,
            candidate_preds=predicates
        )

        encoded_thoughts_prompt = tokenizer(
            thoughts_query, return_tensors='pt'
        )['input_ids'].to(device)

        target_tokens = tokenizer.encode(
                        path[i], 
                        add_special_tokens=False, 
                        return_tensors='pt'
                    ).to(device)

        logp, logeosprob = get_target_tokens_logp(
                    model, 
                    encoded_prompt=encoded_thoughts_prompt.repeat(
                                    len(target_tokens), 1
                                    ), 
                    target_tokens=target_tokens,
                    eos_token_id= tokenizer.eos_token_id, 
                    use_cache=use_cache
                    )
        
        logp_sum += logp
    ## TODO ADD termiating probability to transition 
    return logp_sum
