example_0 = """
Example: Given Events 0, 1

We observe:
1. Event 0 is activated at time 0.4
2. Event 1 is activated at time 0.5

Since Event 1 (t=0.5) happens right after Event 0 (t=0.4)
the most likely event (choose from event list: 1) to cause Event 1 is Event 0 
"""

example_1 = """
Example: Given Events 0, 1, 2 

We observe:
1. Event 1 is activated at time 0.2 0.3
2. Event 2 is activated at time 0.3 0.4
3. Event 0 is activated at time 0.5

Since Event 0 (t=0.5) happens right after Event 2 (t=0.4)
the most likely event (choose from event list: 1, 2) to cause Event 0 is Event 2 
"""

example_2 = """
Example: Given Events 0, 1, 2

We observe:
1. Event 0 is activated at time 0.2, 0.3, 0.5
2. Event 1 is activated at time 0.5, 0.6
3. Event 2 is activated at time 0.1, 0.4

Since Event 1 (t=0.6) happens right after Event 0 (t=0.5)
the most likely event (choose from event list: 0, 2) to cause Event 1 is Event 0 
"""