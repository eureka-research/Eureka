We have a description of a ball's position, and we want you to turn that into the corresponding program by calling the following functions:

```
def set_ball_pos_parameters(pos_x, pos_y, pos_z)
```
pos_x: Where the ball should go in the x axis.
pos_y: Where the ball should go in the y axis.
pos_z: Where the ball should go in the z axis.

```
def set_ball_vel_parameters(vel_x, vel_y, vel_z)
```
vel_x: The ball's desired velocity along the x axis.
vel_y: The ball's desired velocity along the y axis.
vel_z: The ball's desired velocity along the z axis.

Example plan:
The x position of the ball should be 0.0 meters.
The y position of the ball should be 0.0 meters.
The z position of the ball should be 0.0 meters.
The x velocity of the ball should be 1.0 meters per second.
The y velocity of the ball should be 2.0 meters per second.
The z velocity of the ball should be 3.0 meters per second.

Example answer code:
```
set_ball_pos_parameters(0.0, 0.0, 0.0)
set_ball_vel_parameters(1.0, 2.0, 3.0)
```

Remember:
1. Always format the code in code blocks.
2. Do not invent new functions or classes. The only allowed functions you can call are the ones listed above. Do not leave unimplemented code blocks in your response.
3. The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
4. If you are not sure what value to use, just use your best judge. Do not use None for anything.
5. Do not calculate the position or direction of any object (except for the ones provided above). Just use a number directly based on your best guess.
6. Do not provide an implementation for any of the provided functions.
7. Do not wrap your generated code within a function.