We have two dexterous manipulators (shadow hands) and we want you to help plan how it should move to perform tasks using the following template:

[start of description]
object1={CHOICE: <INSERT OBJECTS HERE>} should be {CHOICE: close to, far from} object2={CHOICE: <INSERT OBJECTS HERE>, nothing}.
[optional] object3={CHOICE: <INSERT OBJECTS HERE>} should be {CHOICE: close to, far from} object4={CHOICE: <INSERT OBJECTS HERE>, nothing}.
[optional] object1 needs to have a rotation orientation similar to object2.
[optional] object3 needs to have a rotation orientation similar to object4.
<INSERT OPTIONAL HAND DESCRIPTIONS HERE>
[optional] doors needs to be {CHOICE: open, closed} {CHOICE: inward, outward}.
[optional] scissor needs to be opened to [NUM: 0.0] radians.
[optional] block2 needs to be stacked on top of block1.
[end of description]

Rules:
1. If you see phrases like [NUM: default_value], replace the entire phrase with a numerical value.
2. If you see phrases like {CHOICE: choice1, choice2, ...}, it means you should replace the entire
phrase with one of the choices listed.
3. If you see [optional], it means you only add that line if necessary for the task, otherwise remove that line.
4. The environment contains <INSERT OBJECTS HERE>. Do not invent new objects not listed here.
5. I will tell you a behavior/skill/task that I want the manipulator to perform and you will provide the full plan, even if you may only need to change a few lines. Always start the description with [start of plan] and end it with [end of plan].
6. You can assume that the hands are capable of doing anything, even for the most challenging task.
7. Your plan should be as close to the provided template as possible. Do not include additional details.