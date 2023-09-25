We have a plan of a robot arm with palm to manipulate objects and we want you to turn that into the corresponding program with following functions:

```
def set_min_l2_distance_reward(name_obj_A, name_obj_B)
```
This term sets a reward for minimizing l2 distance between name_obj_A and name_obj_B so they get closer to each other.
name_obj_A and name_obj_B are selected from [<INSERT FIELDS HERE>].

```
def set_max_l2_distance_reward(name_obj_A, name_obj_B)
```
This term sets a reward for maximizing l2 distance between name_obj_A and name_obj_B so they get closer to each other.
name_obj_A and name_obj_B are selected from [<INSERT FIELDS HERE>].

```
def set_obj_orientation_reward(name_obj_A, name_obj_B)
```
This term encourages the orientation of name_obj_A to be close to the orientation of name_obj_B. name_obj_A and name_obj_B are selected from [<INSERT ORIENTATION FIELDS HERE>].

Example plan:
object1=object1 should be close to object2=object1_goal.
object1 needs to have a rotation orientation similar to object2.
To perform this task, the left manipulator's palm should move close to object1.

Example answer code:
```
set_min_l2_distance_reward("object1", "object1_goal")
set_min_l2_distance_reward("object1", "left_palm")
set_obj_orientation_reward("object1", "object1_goal")
```

Remember:
1. Always format the code in code blocks.
2. Do not wrap your code in a function. Your output should only consist of function calls like the example above.
3. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above, and do not implement them. Do not leave unimplemented code blocks in your response.
4. The only allowed library is numpy. Do not import or use any other library.
5. If you are not sure what value to use, just use your best judge. Do not use None for anything.
6. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
7. You do not need to make the robot do extra things not mentioned in the plan such as stopping the robot.