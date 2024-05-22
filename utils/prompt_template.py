llama_prompt = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
"""

prompt_thoughts_example = """
Example: Given events 0, 1, 2

We observe Event 0 is activated at time 0.2, 0.3, 0.5
We observe Event 1 is activated at time 0.5, 0.6
We observe Event 2 is activated at time 0.7, 0.8

You need to list all possible events that cause Event 2 to be activated.

Answer: 0, 1

Explanation:
0 is activated at 0.3, 2 is activated at 0.7, 0.3 is samller than 0.7
1 is activated at 0.5, 2 is activated at 0.8, 0.5 is samller than 0.8
"""

prompt_observation = """
{task_instruction} 
{examples}

Now you have Event {total_events}

We have the observations:
{events_history}
"""

prompt_backward_reasoning = """
I want you to do the reasoning over social events.
Given event list: {total_events}

We have the observations:
{events_history}

If the activation time of one event happens before Event {head_event}, it means that event could have caused Event {head_event} to be activated. 
If the activation time of one event do not happens before Event {head_event}, it means that event cannot cause the other event to be activated. 

Using this logic and based on the previous observation, You need to reason all possible events from above that can cause Event {head_event} to be activated.
Start your answer from the most confident one and stop if you cannot find any other events.

then, the most likely event (choosen from event list : {possible_events}) to cause Event {head_event} is Event
"""


prompt_sols_with_rules = """
I want you to perform inference over social events.
{examples}
Now you have event: {total_events} 
and possible rules:
{rationales}
We have the following observation (events can occur more than once):
{events_history}
then, the most likely event (choosen from event list : {possible_events}) to happen after {time} is Event
"""


# prompt_thoughts = """
# If the activation time of one event happens before Event {head_event}, it means that event could have caused Event {head_event} to be activated. 
# If the activation time of one event do not happens before Event {head_event}, it means that event cannot cause the other event to be activated. 

# Using this logic and based on the previous observation, You need to reason all possible events from above that can cause Event {head_event} to be activated.
# Start your answer from the most confident one and stop if you cannot find any other events.

# then, the most likely event (choosen from event list : {possible_events}) to cause Event {head_event} is Event
# """

prompt_thoughts = """
then, the most likely event (choosen from event list : {possible_events}) to cause Event {head_event} is Event
"""

prompt_thought_prior = """
If the activation time of one event happens before another event, it means that event could have caused another to be activated. 
If the activation time of one event do not happens before another event, it means that event cannot cause the other event to be activated. 

Using this logic and based on the previous observation, We can induce
"""

prompt_sol = """
Let's think step by step, since we have:
{rationales}
then, the most likely event (choosen from event list : {possible_events}) to happen after {time} is Event 
"""

