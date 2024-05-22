import math
import itertools

import numpy as np

PAD = -1

def softplus(x, s):
    return s * np.log(1. + np.exp(x / s))

def weighted_combinations(combinations : list):
    """Find the weighted mean of all satisfying combinations"""
    return max(0, np.sum([comb[-1] for comb in combinations]))

def find_event_combinations_with_weight_backward(
        event_histories : list, 
        sequence : list,
        tolerance=0.3,
        beta=1.0,
        index=0, 
        current_time=np.inf, 
        current_combination=[], 
        current_weight=1.0
    ):
    """Recursively Find all possible weighted combination of events weighted by time delay followed the rule in list
    event_histories : [(event_time, event_type)]
    sequence : an ordered sequence to store rule
    """

    if index < 0:
        # Base case: if the end of the sequence is reached, return the current combination and weight
        return [(list(current_combination), current_weight)]

    combinations = []

    current_sequence = sequence[index]
    if isinstance(current_sequence, list):  # Handling a group of events, e.g., [0, 2]
        # Generate all unique pairs of events within the group

        events_times = [[e_time for e_time, e_type in event_histories if e_type == event] for event in current_sequence ]


        for times in itertools.product(*events_times):
            
            valid_times = [times[i] for i in range(len(times)) if any(abs(times[i] - times[j]) <= tolerance for j in range(len(times)) if i != j)]
            if valid_times and all(time < current_time for time in valid_times):
                new_time = min(valid_times)
                time_diff = (current_time - new_time) if current_time != np.inf else 0
                rule_delay = (event_histories[-1][0] - current_time) if current_time != np.inf else 0
                new_weight = current_weight * math.exp(- beta * time_diff - beta * rule_delay) 
                new_combination = current_combination + [(times[i],current_sequence[i]) for i in range(len(current_sequence))]
                combinations.extend(find_event_combinations_with_weight_backward(
                    event_histories, sequence, tolerance, beta, index - 1, new_time, new_combination, new_weight))
    else:
        event_id = sequence[index] 
        for idx, (e_time, e_type) in enumerate(event_histories[::-1]):
            if e_type == event_id and e_time < current_time:
                # Calculate the weight for this event
                time_diff = (current_time - e_time) if current_time != np.inf else 0
                rule_delay = (event_histories[-1][0] - current_time) if current_time != np.inf else 0
                new_weight = current_weight * math.exp(-beta* time_diff - beta * rule_delay)

                # Create a new combination including this event
                new_combination = current_combination + [(e_time, e_type)]
                combinations.extend(find_event_combinations_with_weight_backward(
                    event_histories, sequence, tolerance, beta, index - 1, e_time, new_combination, new_weight))

    return combinations



def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst.tolist() + [PAD] * (max_len - len(inst))
        for inst in insts])

    return batch_seq


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst.tolist() + [PAD] * (max_len - len(inst))
        for inst in insts])

    return batch_seq


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    
    time, event_type, train_y = list(zip(*insts))

    time = pad_time(time)
    event_type = pad_type(event_type)
    aligned_train_x = np.stack([time, event_type], axis=-1)
    return aligned_train_x, train_y

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    return seq.__ne__(PAD)

def array_2_list(seq : np.ndarray) -> list:
    """Map the aligned np.ndarray to unaligned sequences"""

    return [x[get_non_pad_mask(x)].reshape(-1, 2) for x in seq]


def get_Xt(X, t)-> list:
    """Return the event histories before time t"""
    return [x[x[:,0]<=t] for x in X]