example_0 = """
Example: Given Events 0, 1 and possible rules:

1. Event 1 <-- (Event 0) and (Time of Event 1 after Time of Event 0)

We have the following observation (events can occur more than once):
1. Event 0 is activated at time 0.4

since Event 0 is activated,
then, the most likely event (choose from event list: 0, 1) to happen after 0.4 is Event 1
"""

example_1 = """
Example: Given Events 0, 1, 2 and possible rules:

1. Event 0 <-- (Event 1) and (Time of Event 0 after Time of Event 1), 
2. Event 0 <-- (Event 2) and (Time of Event 0 after Time of Event 2)

We have the following observation (events can occur more than once):
1. Event 1 is activated at time 0.2

since Event 1 is activated,
then, the most likely event (choosen from event list : 0, 1, 2) to happen after 0.2 is Event 0
"""

example_2 = """
Example: Given Events 0, 1, 2 and possible rules:

1. Event 2 <-- (Event 1) and (Event 0) and (Time of Event 2 after Time of Event 1) and (Time of Event 1 after Event 0)

We have the following observation (events can occur more than once):
1. Event 0 is activated at time 0.2, 0.3, 0.5
2. Event 1 is activated at time 0.5, 0.6
3. Event 2 is activated at time 0.1, 0.4

since Event 1 and Event 0 is activated, Event 1 happens after Event 0, 
then, the most likely event (choosen from event list : 0, 1, 2) to happen after 0.8 is Event 2
"""
