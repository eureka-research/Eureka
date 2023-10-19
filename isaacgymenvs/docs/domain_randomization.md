Domain Randomization
====================

Overview
--------

We sometimes need our reinforcement learning agents to be robust to
different physics than they are trained with, such as when attempting a
sim2real policy transfer. Using domain randomization, we repeatedly
randomize the simulation dynamics during training in order to learn a
good policy under a wide range of physical parameters.

IsaacGymEnvs supports "on the fly" domain randomization, allowing dynamics
to be changed when resetting the environment, but without requiring
reloading of assets. This allows us to efficiently apply domain
randomizations without common overheads like re-parsing asset files.
Domain randomization must take place at environment reset time, as some
environment properties are reset when applying randomizations at the
physics simulation level.

We provide two interfaces to add domain randomization to your `isaacgymenvs`
tasks:

1.  Adding domain randomization parameters to your task's YAML config
2.  Directly calling the `apply_randomizations` class method

Underneath both interfaces is a nested dictionary that allows you to
fully specify which parameters to randomize, what distribution to sample
for each parameter, and an option to schedule when the randomizations
are applied or anneal the range over time. We will first discuss all the
"knobs and dials" you can tune in this dictionary, and then how to
incorporate either of the interfaces within your tasks.

Domain Randomization Dictionary
-------------------------------

We will first explain what can be randomized in the scene and the
sampling distributions and schedulers available. There are four main
parameter groups that support randomization. They are:

-   `observations`
    :   -   Add noise directly to the agent observations

-   `actions`
    :   -   Add noise directly to the agent actions

-   `sim_params`
    :   -   Add noise to physical parameters defined for the entire
            scene, such as `gravity`

-   `actor_params`
    :   -   Add noise to properties belonging to your actors, such as
            the `dof_properties` of a ShadowHand

For each parameter you wish to randomize, you can specify the following
settings:

-   `distribution`
    :   -   The distribution to generate a sample `x` from.
        -   Choices: `uniform`, `loguniform`, `gaussian`.
            :   -   `x ~ unif(a, b)`
                -   `x ~ exp(unif(log(a), log(b)))`
                -   `x ~ normal(a, b)`

        -   Parameters `a` and `b` are defined by the `range` setting.

-   `range`
    :   -   Specified as tuple `[a, b]` of real numbers.
        -   For `uniform` and `loguniform` distributions, `a` and `b`
            are the lower and upper bounds.
        -   For `gaussian`, `a` is the distribution mean and `b` is the
            variance.

-   `operation`
    :   -   Defines how the generated sample `x` will be applied to the
            original simulation parameter.
        -   Choices: `additive`, `scaling`
            :   -   For `additive` noise, add the sample to the original
                    value.
                -   For `scaling` noise, multiply the original value by
                    the sample.

-   `schedule`
    :   -   Optional parameter to specify how to change the
            randomization distribution over time
        -   Choices: `constant`, `linear`
            :   -   For a `constant` schedule, randomizations are only
                    applied after `schedule_steps` frames.
                -   For a `linear` schedule, linearly interpolate
                    between no randomization and maximum randomization
                    as defined by your `range`.

-   `schedule_steps`
    :   -   Integer frame count used in `schedule` feature

-   `setup_only`
    :   -   Specifies whether the parameter is to be randomized during setup only. Defaults to `False`
        -   If set to `True`, the parameter will not be randomized or set during simulation
        -   `Mass` and `Scale` must have this set to `True` - the GPU pipeline API does not currently support changing these properties at runtime. See Programming/Physics documentation for Isaac Gym for more details
        -   Requires making a call to `apply_randomization` before simulation begins (i.e. inside `create_sim`)

We additionally can define a `frequency` parameter that will specify how
often (in number of environment steps) to wait before applying the next
randomization. Observation and action noise is randomized every frame,
but the range of randomization is updated per the schedule only every
`frequency` environment steps.

YAML Interface
--------------

Now that we know what options are available for domain randomization,
let's put it all together in the YAML config. In your isaacgymenvs/cfg/task yaml
file, you can specify your domain randomization parameters under the
`task` key. First, we turn on domain randomization by setting
`randomize` to `True`:

    task:
        randomize: True
        randomization_params:
            ...

Next, we will define our parameters under the `randomization_params`
keys. Here you can see how we used the previous settings to define some
randomization parameters for a ShadowHand cube manipulation task:

    randomization_params:
        frequency: 600  # Define how many frames between generating new randomizations
        observations:
            range: [0, .05]
            operation: "additive"
            distribution: "uniform"
            schedule: "constant"  # turn on noise after `schedule_steps` num steps
            schedule_steps: 5000
        actions:
            range: [0., .05]
            operation: "additive"
            distribution: "uniform"
            schedule: "linear"  # linearly interpolate between 0 randomization and full range
            schedule_steps: 5000
        sim_params: 
            gravity:
                range: [0, 0.4]
                operation: "additive"
                distribution: "uniform"
        actor_params:
            hand:
                color: True
                dof_properties:
                    upper:
                        range: [0, 0.15]
                        operation: "additive"
                        distribution: "uniform"
            cube:
                rigid_body_properties:
                    mass: 
                        range: [0.5, 1.5]
                        operation: "scaling"
                        distribution: "uniform"
                        setup_only: True

Note how we structured the `actor_params` randomizations. When creating
actors using `gym.create_actor`, you have the option to specify a name
for your actor. We figure out which randomizations to apply to actors
based on this name option. **To use domain randomization, your agents
must have the same name in** `create_actor` **and in the randomization
YAML**. In our case, we wish to randomize all ShadowHand instances the
same way, so we will name all our ShadowHand actors as `hand`. Depending
on the asset, you have access to randomize `rigid_body_properties`,
`rigid_shape_properties`, `dof_properties`, and `tendon_properties`. We
also include an option to set the `color` of each rigid body in an actor
(mostly for debugging purposes), but do not support extensive visual
randomizations (like lighting and camera directions) currently. The
exact properties available are listed as follows.

**rigid\_body\_properties**:

        (float) mass # mass value, in kg
        (float) invMass # Inverse of mass value.

**rigid\_shape\_properties**:

    (float) friction # Coefficient of static friction. Value should be equal or greater than zero.
    (float) rolling_friction # Coefficient of rolling friction.
    (float) torsion_friction # Coefficient of torsion friction.
    (float) restitution # Coefficient of restitution. It's the ratio of the final to initial velocity after the rigid body collides. Range: [0,1]
    (float) compliance # Coefficient of compliance. Determines how compliant the shape is. The smaller the value, the stronger the material will hold its shape. Value should be greater or equal to zero.
    (float) thickness # How far objects should come to rest from the surface of this body

**dof\_properties**:

    (float) lower # lower limit of DOF. In radians or meters
    (float) upper \# upper limit of DOF. In radians or meters
    (float) velocity \# Maximum velocity of DOF. In Radians/s, or m/s
    (float) effort \# Maximum effort of DOF. in N or Nm.
    (float) stiffness \# DOF stiffness.    
    (float) damping \# DOF damping.    
    (float) friction \# DOF friction coefficient, a generalized friction force is calculated as DOF force multiplied by friction.
    (float) armature \# DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia.

**tendon\_properties**:

        (float) stiffness # Tendon spring stiffness
        (float) damping # Tendon and limit damping. Applies to both tendon and limit spring-damper dynamics.
        (float) fixed_spring_rest_length # Fixed tendon spring rest length. When tendon length = springRestLength the tendon spring force is equal to zero
        (float) fixed_lower_limit # Fixed tendon length lower limit
        (float) fixed_upper_limit # Fixed tendon length upper limit

To actually apply randomizations during training, you will need to have
a copy of the params available in your task class instance, and to call
`self.apply_randomizations`. The easiest way to do is to instantiate a
dictionary with the parameters in your Task's `__init__` call:

    self.randomization_params = self.cfg["task"]["randomization_params"]

We also recommend that you call `self.apply_randomizations` once in your
`create_sim()` code to do an initial randomization pass before simulation 
starts. This is required for randomizing `mass` or `scale` properties.

Supporting scheduled randomization also requires adding an additional
line of code to your `post_physics_step()` code to update how far along
in randomization scheduling each environment is - this is stored in the
`randomize_buf` tensor in the base class:

    def post_physics_step(self):
        self.randomize_buf += 1

Finally, add a call to `apply_randomizations` during the reset portion
of the training loop. The function takes as arguments a domain
randomization dictionary:

    def reset(self, env_ids):
        self.apply_randomizations(self.randomization_params)
                ...

Only environments that are in the reset buffer and which have exceeded
the specified `frequency` time-steps since last randomized will have
new randomizations applied.

Custom domain randomizations
----------------------------

**Custom randomizations via a class method**:

Provided your task inherits from our `VecTask` class, you have great
flexibility in choosing when to randomize and what distributions to
sample, and can even change the entire domain randomization dictionary
at every call to `apply_randomizations` if you wish. By using your own
logic to generate these dictionaries, our current framework can be
easily extended to use more intelligent algorithms for domain
randomization, such as ADR or BayesSim.


Automatic Domain Randomisation 
------------------------------

Our [DeXtreme](https://dextreme.org) work brings Automatic Domain Randomisation (ADR) into Isaac Gym. Since, the simulator is built on vectorising environments on the GPU, our ADR naturally comes with vectorised implementation. Note that we have only tested ADR for DeXtreme environments mentioned in [dextreme.md](dextreme.md) and we are working towards bringing ADR and DeXtreme to [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

**Background**

ADR was first introduced in [OpenAI 2019 et. al](https://arxiv.org/abs/1910.07113). We develop the vectorised version of this and use that to train our policies in sim and transfer to the real world. Our experiments reaffirm that ADR imbues robustness to the policies closing the sim-to-real gap significantly leading to better performance in the real world compared to traiditional manually tuned domain randomisation.

Hand-tuning the randomisation ranges (_e.g._ means and stds of the distributions) of parameters can be onerous and may result in policies that lack adaptability, even for slight variations in parameters outside of the originally defined ranges. ADR starts with small ranges and automatically adjusts them gradually to keep them as wide as possible while keeping the policy performance above a certain threshold. The policies trained with ADR exhibit significant robustness to various perturbations and parameter ranges and improved sim-to-real transfer. Additionally, since the ranges are adjusted gradually, it also provides a natural curriculum for the policy to absorb the large diverity thrown at it.

Each parameter that we wish to randomise with ADR is modelled with uniform distribution `U(p_lo, p_hi)` where `p_lo` and `p_hi` are the lower and the upper limit of the range respectively. At each step, a parameter is randomy chosen and its value set to either the lower or upper limit keeping the other parameters with their ranges unchanged. This randomly chosen parameter's range is updated based on its performance. A small fraction of the overall environments (40% in our [DeXtreme](https://dextreme.org) work) is used to evaluate the performance. Based on the performance, either the range shrinks or expands. A visualisation from the DeXtreme paper is shown below: 

![ADR](https://user-images.githubusercontent.com/686480/228732516-2d70870d-828c-4934-a3c2-17b989683a6d.png)

If the parameter value was set to the lower limit, then a decrease in performance, measured by performance threshold `t_l`, dicatates reducing the range of the parameter (shown in (a) in the image) by increasing the lower limit value by a small delta. Conversely, if the performance is increased, measured by performance threshold, `t_h`, the lower limit is decreased (shown in (c) in the image) leading to expanding the overall range.

Similarly, if the parameter value was set to the upper limit, then an increase in performance, measured by performance threshold `t_h`, expands the range (shown in (b) in the image) by increasing the upper limit value by a small delta. However, if the performance is decreased, measured by performance threshold, `t_l`, the upper limit is decreased (shown in (d) in the image) leading to shrinking the overall range.

**Implementation**

The ADR implementation resides in [adr_vec_task.py](../isaacgymenvs/tasks/dextreme/adr_vec_task.py) located in `isaacgymenvs/tasks/dextreme` folder. The `ADRVecTask` inherits much of the `VecTask` functionality and an additional class to denote the state of the environment when evaluating the performance 

```
class RolloutWorkerModes:
    ADR_ROLLOUT  = 0 # rollout with current ADR params
    ADR_BOUNDARY = 1 # rollout with params on boundaries of ADR, used to decide whether to expand ranges
```

Since ADR needs to have the evaluation in the loop to benchmark the performance and adjust the ranges consequently, some fraction of the environments are dedicated to the evaluation denoted by `ADR_BOUNDARY`. Rest of the environments continue to use the unchanged ranges and are denoted by `ADR_ROLLOUT`.

The `apply_randomisation` has additional arguments this time `randomise_buf`, `adr_objective` and `randomisation_callback`. The variable `randomise_buf` enables selective randomisation of some environments while keeping others unchanged, `adr_objective` is the number of consecutive successes and `randomisation_callback` allows using any callbacks for randomisation from the `ADRDextreme` class.

YAML Interface 
--------------

The YAML file interface now has additional `adr` key where we need to set the appropriate variables and it looks like the following:

```
adr:

    use_adr: True

    # set to false to not do update ADR ranges. 
    # useful for evaluation or training a base policy
    update_adr_ranges: True 
    clear_other_queues: False

    # if set, boundary sampling and performance eval will occur at (bound + delta) instead of at bound.
    adr_extended_boundary_sample: False

    worker_adr_boundary_fraction: 0.4 # fraction of workers dedicated 
    to measuring perf of ends of ADR ranges to update the ranges

    adr_queue_threshold_length: 256

    adr_objective_threshold_low: 5
    adr_objective_threshold_high: 20

    adr_rollout_perf_alpha: 0.99
    adr_load_from_checkpoint: false

    params:
      ### Hand Properties
      hand_damping:
        range_path: actor_params.hand.dof_properties.damping.range
        init_range: [0.5, 2.0]
        limits: [0.01, 20.0]
        delta: 0.01
        delta_style: 'additive'
        ....
```

Lets unpack the variables here and go over them one by one:

- `use_adr`: This flag enables ADR. 
- `update_adr_ranges`: This flag when set to `True` ensures that the ranges of the parameters are updated.
- `clear_other_queues`: This means that for when evaluating parameter A, whether we want to clear the queue for parameter B. More information on the queue is provided for `adr_queue_threshold_length` below.
-  `adr_extended_boundary_sample`: We test the performance at either the boundary of the parameter limits of boundary + delta. When this flag is set to `True`, the performance evaluation of the parameter is doing on boundary + delta instead of boundary.
- `worker_adr_boundary_fraction`: For the evaluation, certain fraction of the overall environments are chosen and this variable allows setting that fraction. 
- `adr_queue_threshold_length`: The performance is evaluated periodically and stored in a queue and averaged. This variable allows choosing the length of the queue so that statistics are computed over a sufficiently large window. We do not want to rely on policy achieving the thresholds by chance; we want it to maintain the peaks for a while. Therefore, a queue allows logging statistics over a given time frame to be sure that its performing above the threshold.
- `adr_objective_threshold_low`: This is the `t_l` threshold mentioned in the **Background** section above. Also shown in the image.
- `adr_objective_threshold_high`: This is the `t_h` threshold as mentioned above in the image.
- `adr_rollout_perf_alpha`: This is the smoothing factor used to compute the performance.
- `adr_load_from_checkpoint`: The saved checkpoints also contain the ADR optimised ranges. Therefore, if you want to load up those ranges for future post-hoc evaluation, you should set this to `True`. If set to `False`, it will only load the ranges from the YAML file and not update them from the checkpoint.

Additionally, as you may have noticed, each parameter now also comes with `limit` and `delta` variables. The variable `limits` refers to the complete range within which the parameter is permitted to move, while `delta` represents the incremental change that the parameter can undergo with each ADR update.  


