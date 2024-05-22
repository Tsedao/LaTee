import numpy as np
import itertools
import tqdm
import pdb
from tqdm import tqdm
from utils.data_utils import (
    find_event_combinations_with_weight_backward,
    weighted_combinations,
)

def energy(y : int , event_history : list, rule : list ) -> float:
    """Return the energy for class y at rule x based on the event history"""
    ## check whether y is head predicates or not
    if y != rule[-1]:
        return 0 ## avoid zero energy
    
    combs = find_event_combinations_with_weight_backward(
        event_histories=event_history, 
        sequence=rule, 
        index= len(rule) - 1,
        beta=2.0,
        )
    energy = weighted_combinations(combs)
    return energy


def rules_mapping(head_predicates, complete_predicates,max_len=4):
    rule_forward_mapping = {}
    rule_inverse_mapping = {}

    # Function to generate combinations starting with a head predicate
    def generate_combinations(head, all_predicates, max_len=3):
        remaining_predicates = list(set(all_predicates) - {head})
        for length in range(0, max_len + 1):
            for combo in itertools.combinations(remaining_predicates, length):
                for perm in itertools.permutations(combo):
                    yield perm + (head,)

    # Generate all possible combinations
    combinations = []
    for head in head_predicates:
        combinations.extend(generate_combinations(head, complete_predicates, max_len-1))

    
    # Create forward and inverse mappings
    print(len(combinations))
    for index, combo in tqdm(enumerate(combinations), desc='initialize rule space'):
        rule_forward_mapping[combo] = index
        rule_inverse_mapping[index] = combo
    # cnt = 0
    # for c_pred in complete_predicates:
    #     for h_pred in complete_predicates:
    #         if c_pred != h_pred:
    #             preds_mapping[(c_pred, h_pred)] = cnt
    #             inverse_mapping[cnt] = (c_pred, h_pred)
    #             cnt += 1
    # return preds_mapping, inverse_mapping

    return rule_forward_mapping, rule_inverse_mapping

def rule_padding2d(rules :list[list[int]], rule_mapping:dict):
    padding = np.zeros(len(rule_mapping))
    for rule in rules:
        assert len(rule) == 2
        idx = rule_mapping[tuple(rule)]
        padding[idx] = 1
    return padding 


def get_logic_features(
        X : list[list], 
        predicates : list, 
        rules : list,
        rule_mapping : dict
    ) -> np.ndarray:
    """Return the logic features based on event history X on all head predicates
    Inputs:
        X: list of event histories
        predicates: list of head/ body preds
        rules : list of rule
        rule_mapping : mapping the rule to index
    Returns:
        logic features (np.ndarray): [BS, NUM_PREDS, RULES] 
    """
    # print(rule_mapping, len(rule_mapping))
    features = np.zeros(shape=(len(X), len(predicates), len(rule_mapping)), dtype=np.float64)
    # print(features.shape)
    for i in range(len(X)):
        for _, rule in enumerate(rules):
            for h_idx, head_pred in enumerate(predicates):
                feature = energy(y=head_pred, event_history=X[i], rule=rule)
                r_idx = rule_mapping[tuple(rule)]
                try:
                    features[i,h_idx, r_idx] = feature
                except KeyError:
                    print(f'rule {rule} not found in rule dict {rule_mapping}' )
                    raise KeyError
    # print("logical features",features)
    return features