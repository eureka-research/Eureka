Reproducibility and Determinism
===============================

Seeds
-----

To achieve deterministic behaviour on multiple training runs, a seed
value can be set in the training config file for each task. This will potentially
allow for individual runs of the same task to be deterministic when
executed on the same machine and system setup. Alternatively, a seed can
also be set via command line argument `seed=<seed>` to override any
settings in config files. If no seed is specified in either config files
or command line arguments, we default to generating a random seed. In
that case, individual runs of the same task should not be expected to be
deterministic. For convenience, we also support setting `seed=-1` to
generate a random seed, which will override any seed values set in
config files. By default, we have explicitly set all seed values in
config files to be 42.

PyTorch Deterministic Training
------------------------------

We also include a `torch_deterministic` argument for uses when running RL
training. Enabling this flag (passing `torch_deterministic=True`) will
apply additional settings to PyTorch that can force the usage of deterministic 
algorithms in PyTorch, but may also negatively impact run-time performance. 
For more details regarding PyTorch reproducibility, refer to
<https://pytorch.org/docs/stable/notes/randomness.html>. If both
`torch_deterministic=True` and `seed=-1` are set, the seed value will be
fixed to 42.

Note that in PyTorch version 1.9 and 1.9.1 there appear to be bugs affecting
the `torch_deterministic` setting, and using this mode will result in a crash,
though in our testing we did not notice determinacy issues arising from not 
setting this flag.

Runtime Simulation Changes / Domain Randomization
-------------------------------------------------

Note that using a fixed seed value will only **potentially** allow for deterministic 
behavior. Due to GPU work scheduling, it is possible that runtime changes to 
simulation parameters can alter the order in which operations take place, as 
environment updates can happen while the GPU is doing other work. Because of the nature 
of floating point numeric storage, any alteration of execution ordering can 
cause small changes in the least significant bits of output data, leading
to divergent execution over the simulation of thousands of environments and
simulation frames.

As an example of this, runtime domain randomization of object scales or masses 
are known to cause both determinacy and simulation issues when running on the GPU 
due to the way those parameters are passed from CPU to GPU in lower level APIs. By 
default, in examples that use Domain Randomization, we use the `setup_only` flag to only 
randomize scales and masses once across all environments before simulation starts. 

At this time, we do not believe that other domain randomizations offered by this
framework cause issues with deterministic execution when running GPU simulation, 
but directly manipulating other simulation parameters outside of the Isaac Gym tensor 
APIs may induce similar issues.

CPU MultiThreaded Determinism
-----------------------------

We are also aware of one environment (Humanoid) that does not train deterministically
when simulated on CPU with multiple PhysX worker threads. Similar to GPU determinism
issues, this is likely due to subtle simulation operation ordering issues, and additional
effort will be needed to enforce synchronization between threads.

We have not observed similar issues when using CPU simulation with other examples, or
when restricting CPU simulation to a single thread