prompt_thoughts_template="""
Inputs: Given events A, B, C

Event A is recorded at time 0.2, 
- We observe Event A is activated at time 0.2 
- We also observe Event A is not activated at time  

Event B is recorded at time 0.5, 
- We observe Event B is activated at time 0.5 
- We also observe Event B is not activated at time  

Event C is recorded at time 0.4 and 0.7, 
- We observe Event C is activated at time 0.7 
- We also observe Event C is not activated at time 0.4

Question: What possible event will cause Event C be activated?
List all possible events to from: A, B and your confidence levels (certain/high/medium/low), using the format "#EVENT NAME# (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct event.

Answers: 

- Event B (high). Reason: C is not activated at time 0.4, however, when B is activated at time 0.5, C later is activated at time 0.7. So it is likely B causes C to be activated.
- Event A (low).  Reason: A is activated at time 0.2, however, C is not activated at time 0.4. So A can not cause C to be activated.
"""


prompt_thoughts="""
I want you to do the reasoning over social events.
Example: 
Given events 0, 1, 2

We observe Event 0 is activated at time 0.2, 0.3, 0.5
We observe Event 1 is activated at time 0.5, 0.6
We observe Event 2 is activated at time 0.7, 0.8

You need to list all possible events and confidence levels (certain/high/medium/low) that cause Event 2 to be activated.
Using the format "Event NAME (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct event.

Answer: 

- Event 0 (high). Reason : 0 is activated at 0.3, 2 is activated at 0.7, 0.3 is samller than 0.7
- Event 1 (medium). Reason: 1 is activated at 0.5, 2 is activated at 0.8, 0.5 is samller than 0.8


Now Given events {total_events}

{events_history}

Questions: What possible event will cause Event {head_event} be activated?
You have to choose from all possible events from {possible_events} and along with your confidence levels (certain/high/medium/low), using the format "#EVENT NAME# (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct event. 
If you don't know the answer, guess the most possible one.

Answers:
"""

prompt_sol = """
I want you to do the inference over social events.
Example: 
Given events 0, 1, 2

We observe Event 0 is activated at time 0.2, 0.3, 0.5
We observe Event 1 is activated at time 0.5, 0.6
We observe Event 2 is activated at time 0.7, 0.8

what is the most likely event (choosen from event list: 0, 1, 2) to happen after 0.8?

Let's think step by step, since we have:
1. Event 0 starts before Event 2,
2. Event 0 starts before Event 1, Event 1 starts before Event 2,

based on the previous observation and rules, the most likely event to happen after 0.8 is: 
#Event 2#  

Now Given events {total_events}

{events_history}

Let's think step by step, since we have:
{rationales}
then, the most likely event (choosen from event list : {possible_events}) to happen after {time} is (write your answer in #Event NAME# format):
"""

