import numpy as np
import heapq

import pickle 

import matplotlib.pyplot as plt
from utils.data_utils import (
    find_event_combinations_with_weight_backward,
    weighted_combinations,
    softplus
)

class LogicTPPGenerator:
    def __init__(
            self, 
            rules : list[list], 
            rule_weights: list[float],
            head_predicates: list, 
            body_predicates: list,
            head_mu : list,
            body_mu : list,
            alpha = None,
            beta = None,
        ):
        '''params should be of form:
        rules : [[0, 1, 2], [2, 3, 4]]
        We use the formulation in Tranformer Hawkes Process to build the intensity function
        lambda_k = f_{k} (\alpha_{k} t-tj / tj + w_{f} \phi_{f}(x) + b_{k})
        '''
        
        self.data = []
        self.event_queue = []            # min-heap for recent events
        self.rules = rules
        self.rule_weights = rule_weights
        self.head_mu = head_mu
        self.body_mu = body_mu             # fixed
        self.head_predicates = head_predicates
        self.body_predicates = body_predicates
        assert len(self.body_mu) == len(self.body_predicates), f"Length of based intensity does not equal length of head predicates"
        assert len(self.head_mu) == len(self.head_predicates)


        self.mu = self.get_mu(self.head_mu)

        self.dim = len(self.mu)
        self.alpha = np.ones_like(self.mu) if alpha is None else alpha
        self.beta = np.ones_like(self.mu) if beta is None else beta
        
        ## here we assume the maximum lambda for each event as 5
        self.Istar = 10 * len(self.mu)
        self.check_stability()

    def get_mu(self, head_mu):
        """Get the whole mu by appending head_mu and body_mu"""
        mu = np.zeros(shape=len(self.body_mu + head_mu))
        k,j = 0, 0
        for i in range(len(mu)):
            if i in self.head_predicates:
                mu[i] = head_mu[k]
                k += 1
            else:
                mu[i] = self.body_mu[j]
                j += 1
        return mu

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        pass

    def generate_seq(self, horizon, event_nums=40):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Thinning method, with some speedups, noted below'''

        self.data = []  # clear history
        self.event_queue = []

        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), 
                              1, 
                              p=(self.mu / Istar))
        self.data.append([s, n0.item()])
        heapq.heappush(self.event_queue, (s, n0.item()))
        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        while True:
            Istar = self.Istar           

            # rates_max = []
            # for i in range(len(self.head_predicates) + len(self.body_predicates)):
            #     max_rate = 0
            #     for t in np.linspace(s, horizon,50):
            #         max_rate = max(max_rate, self.get_rates(ct=t,d=i, data=self.data))
            #     rates_max.append(max_rate)
            # print(rates_max)
            # Istar = np.sum(rates_max)
            
            # generate new event
            s += np.random.exponential(scale=1./Istar)
            rates = []
            for i in range(len(self.head_predicates) + len(self.body_predicates)):
                # use most recent historical data to calcalue intensity
                rates.append(self.get_rates(ct=s,d=i, data=self.data))
            rates = np.array(rates)
            combined_rates = np.sum(rates)

            # print("Istar:", Istar, "Combined rates:", combined_rates)
            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            u = np.random.uniform(0, self.Istar)
            if u <= combined_rates:
                n0 = np.random.choice(
                    np.arange(self.dim), 1, 
                    p=(rates  / combined_rates)
                )
                # print(rates)
                print(rates  / combined_rates, 'time', s, 'accept', n0.item())
                self.data.append([s, n0.item()])
                heapq.heappush(self.event_queue, (s, n0.item()))
            
            # if past horizon or excede event nums, done
            if s >= horizon or len(self.data) >= event_nums:
                self.data = np.array(self.data)
                self.data = self.data[self.data[:,0] < horizon]
                return self.data
            
            ## remove old events to modelling time-limited influence
            while self.event_queue and len(self.event_queue) > 20:
                heapq.heappop(self.event_queue)
    

    def get_rates(self, ct, d, data):
        # return rate at time ct in dimension d
        if len(data) == 0: return self.mu[d]
        tj = data[-1][0]
        rate =  ( ct - tj ) / tj * self.alpha[d] + self.mu[d]
        for idx, rule in enumerate(self.rules):

            if rule[-1] == d: ## if the head predicate is current dim
                combinations = find_event_combinations_with_weight_backward(
                    event_histories= data,
                    sequence= rule,
                    index= len(rule) - 1,
                    beta = 2
                )
                logic_features = weighted_combinations(combinations=combinations)
                # print(f"rule: {idx}, head: {d}, ct: {ct}, feature: {logic_features}")
                logic_features *= self.rule_weights[idx]

                rate += logic_features
        # return np.exp(rate)
        return np.clip(softplus(rate, s=self.beta[d]),0, 10)
    

    def plot_rates(self, data, horizon=-1):

        if horizon < 0:
            horizon = np.amax(data[:,0])

        f, axarr = plt.subplots(self.dim*2,1, sharex='col', 
                                gridspec_kw = {'height_ratios':sum([[3,1] for i in range(self.dim)],[])}, 
                                figsize=(8,self.dim*2))
        xs = np.linspace(0, horizon, int((horizon/100.)*1000))
        for i in range(self.dim):
            row = i * 2

            # plot rate            
            r = [self.get_rates(ct, i, data=np.array(data)
                    # [np.logical_and(
                    #     np.array(data)[:,0] < ct,
                    #     np.array(data)[:,0] >= max( ct -10, 0),
                    # )]
                    ) for ct in xs]

            axarr[row].plot(xs, r, 'k-')
            axarr[row].set_ylim([-0.01, np.amax(r)+(np.amax(r)/2.)])
            axarr[row].set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
            r = []

            # plot events
            subseq = data[self.data[:,1]==i][:,0]
            axarr[row+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.2)
            axarr[row+1].yaxis.set_visible(False)

            axarr[row+1].set_xlim([0, horizon])

        plt.tight_layout()

if __name__ == "__main__":

    from scipy import stats
    import itertools
    import os
    import argparse
    import random
    from collections import Counter

    # Create the parser
    parser = argparse.ArgumentParser(description='SynDataGenerator')
    parser.add_argument('--rule_nums', type=int, default=5)
    parser.add_argument('--head_nums', type=int, default=3)
    parser.add_argument('--body_nums', type=int, default=2)
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--steps', type=int, default=2000)

    # Parse the arguments
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_rules = args.rule_nums
    num_head_events = args.head_nums
    num_body_events = args.body_nums

    total_events_num = num_head_events + num_body_events

    body_predicates = range(0, num_body_events)
    head_predicates = range(num_body_events , num_body_events + num_head_events)
    all_predicates =  set(body_predicates) | set(head_predicates)

    # Function to generate combinations starting with a head predicate
    def generate_combinations(head, all_predicates, max_len = 3):
        remaining_predicates = list(all_predicates - {head})
        ## truncate the remainning predicates in generation to save time
        remaining_predicates = random.sample(remaining_predicates, 10) 
        for length in range(1, len(remaining_predicates) + 1):
            for combo in itertools.combinations(remaining_predicates, length):
                if len(combo) < max_len:
                    yield  list(combo + (head,))

    # Generate all possible combinations
    combinations = []
    for head in head_predicates:
        combinations.extend(generate_combinations(head, all_predicates))

    rules = random.sample(
        combinations, 
        num_rules
    )

    rule_weights = np.random.normal(size=num_rules).tolist()
    head_mu = np.random.uniform(size=num_head_events).tolist()
    body_mu = np.random.uniform(size=num_body_events).tolist()
    print(f"rules: {rules}")
    print(f"rule weights : {rule_weights}")
    # rules = [
    #     [1, 4],
    #     [2, 4],
    #     [0, 2],
    #     [1, 2],
    #     [0, 3],
    # ]

    # rule_weights = [
    #     0.58, 1.32, 3.87, -1.93, 2.22
    # ]

    # head_predicates = [2, 3, 4]
    # head_mu = [0.3, 0.1, 0.2]

    # body_predicates = [0, 1]
    # body_mu = [0.2, 1.2]

    X_list = []
    y_list = []
    y_pred = []

    horizon = 20
    max_events = 40
    LogicModel = LogicTPPGenerator(
            rules=rules,
            rule_weights=rule_weights,
            head_predicates=head_predicates,
            head_mu= head_mu,
            body_predicates= body_predicates,
            body_mu= body_mu
        )

    for e in range(args.steps):
        data = LogicModel.generate_seq(horizon, event_nums=max_events)
        idices = []
        for i in range(len(data)-10, len(data)):
            event_time, event_type = data[i,0], data[i,1]
            previous_window = horizon * 0.1 ## we only consider the last 10 % events 
            # if event_time > horizon - previous_window:
            if event_type in head_predicates:
                idices.append(i)
        if len(idices) >= 5:
            y = []
            for i in idices:
                y_ = data[i,1]
                y.append(y_)
            X = data[:idices[0],:]

            y_logits = np.zeros(shape=(len(head_predicates)+len(body_predicates)))
            # for idx, j in enumerate(body_predicates):
            #     y_logits[j] = body_mu[idx]

            for i in range(len(rules)):
                logic = weighted_combinations(find_event_combinations_with_weight_backward(
                    X,sequence=rules[i], index=len(rules[i])-1, beta=2))
                head_pred = rules[i][-1]
                y_logits[head_pred] = logic * rule_weights[i] + head_mu[head_predicates.index(head_pred)]
            y_pred.append(np.argmax(y_logits))
            y_true = stats.mode(y)[0]
            ## get the first event belong to the predicate space
            # y_true = y[0]

            X_list.append(X)
            y_list.append(int(y_true))
    
    acc = (y_pred == np.array(y_list)).mean()
    print(f"Ground Truth Acc (Using Ground Truth Rule): {acc:.3f}")
    print(f"Num Samples: {len(X_list)}")
    print(f"Average Event Nums: {sum(len(X_list[i]) for i in range(len(X_list))) / len(X_list):.2f}")

    # Counting the frequency
    freq = Counter(np.array(y_list))

    # Displaying the frequency
    print("Target predicates statistics")
    for number, count in freq.items():
        print(f"Number {number} appears {count} times")

    save_path = f"data/synthetic{total_events_num}"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"synthetic{total_events_num}_x.pkl"), 'wb') as f:
        pickle.dump(X_list, f)

    with open(os.path.join(save_path, f"synthetic{total_events_num}_y.pkl"), 'wb') as f:
        pickle.dump(y_list, f)


    data_config ={
        'head_preds' : list(head_predicates),
        'body_preds' : list(body_predicates),
        'body_mu' : body_mu,
        "head_mu" : head_mu,
        'horizon': horizon,
        'rules': rules,
        'rule_weights': rule_weights,
        'data_seed' : args.seed
    }

    import json
    print(data_config)
    with open(os.path.join(save_path, f"synthetic{total_events_num}_configs.json"), 'w') as f:
        json.dump(data_config, f)