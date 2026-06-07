# OpenArm Pick & Place

RL task for the **OpenArm** robot (7-DOF arm + 2-DOF gripper) in **Isaac Sim 5.0 / Isaac Lab**:
grasp an object (r=1.5 cm, h=8 cm bolt-cylinder) from the table and place it
into a KLT bin. Object and tray positions are randomized on every reset. Training — **PPO**
via `rsl_rl` in **4096 parallel environments**.

![OpenArm Pick & Place demo](pnp.gif)

Registered gym id: `Template-Openarm-Pnp-v0`.

---

## 1. Installation

### 1.1. Prerequisites

- Installed **Isaac Lab** ([guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
  A conda/uv environment is recommended — then `python` sees Isaac Lab directly.
- NVIDIA GPU supported by Isaac Sim (≥ 8 GB VRAM for `num_envs=4096`).
- `git`, `python ≥ 3.10`.

### 1.2. Clone

Clone **next to** `IsaacLab` (not inside it):

```bash
git clone <repo-url> openarm_pnp
cd openarm_pnp
```

### 1.3. Editable install

```bash
# If Isaac Lab is installed in a conda/uv environment:
python -m pip install -e source/openarm_pnp

# If Isaac Lab is installed standalone — use its python:
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/openarm_pnp
```

### 1.4. Verify installation

```bash
python scripts/list_envs.py
# should print a line containing Template-Openarm-Pnp-v0
```

Sanity-check the environment with zero / random actions:

```bash
python scripts/zero_agent.py   --task=Template-Openarm-Pnp-v0
python scripts/random_agent.py --task=Template-Openarm-Pnp-v0
```

---

## 2. Running

### 2.1. Train from scratch

```bash
python scripts/rsl_rl/train.py --task=Template-Openarm-Pnp-v0
```

Checkpoints are saved every 50 iterations to
`logs/rsl_rl/openarm_pnp/<YYYY-MM-DD_HH-MM-SS>/`.

Useful flags:

| Flag | Purpose |
|---|---|
| `--headless` | no GUI (faster) |
| `--num_envs N` | override number of envs |
| `--max_iterations N` | cap the number of iterations |
| `--seed N` | fix the random seed |
| `--video --video_length L --video_interval I` | record training video |

### 2.2. Resume from a checkpoint

```bash
python scripts/rsl_rl/train.py --task=Template-Openarm-Pnp-v0 \
    --resume \
    --load_run=<YYYY-MM-DD_HH-MM-SS> \
    --checkpoint=model_<iter>.pt \
    --max_iterations=300
```

> `--resume` is a **flag with no value**. Do not write `--resume=True`.

### 2.3. Play (evaluate a trained policy)

```bash
python scripts/rsl_rl/play.py --task=Template-Openarm-Pnp-v0 \
    --checkpoint=/absolute/path/to/model_<iter>.pt
```

### 2.4. TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/openarm_pnp
# open http://localhost:6006
```

Key metrics:

| Metric | Healthy | Warning |
|---|---|---|
| `Train/mean_reward` | grows | drops / `inf` |
| `Episode_Reward/success_bonus` | grows | =0 (no successes) |
| `Episode_Termination/screw_placed_success` | → 1.0 | stuck at 0 |
| `Episode_Termination/screw_dropped` | decreases | grows |
| `Mean episode length` | → 50–100 steps | stuck at 250 (time-out) |
| `Policy/mean_noise_std` | 1.0 → 0.3–0.5 | > 2.0 (collapse) |
| `Loss/value_function` | finite | `inf` (policy is dead) |

---

## 3. Environment

### 3.1. Scene

| Object | Type | Position | Size | Mass |
|---|---|---|---|---|
| Table | ThorlabsTable USD | `[0, 0, 0.8]` | default | static |
| Robot | OpenArm unimanual | `[0, 0, 0.8]` | 7 + 2 DOF | — |
| Tray | KLT bin USD (scale 0.75) | `[0.55, 0, 0.85]` ± rand | ~30 cm | **5 kg** (dynamic) |
| Object (cube) | DexCube USD (scale 0.8) | `[0.4, 0, 0.855]` ± rand | 4 cm | default |
| Object (bolt) | CylinderCfg | `[0.4, 0, 0.86]` ± rand | r=1.5 cm, h=8 cm | 50 g |

The tray is **dynamic but heavy** — physics is realistic, but the robot cannot
"push the tray toward the object".

Reset randomization:

```text
object:  Δx ∈ [-0.10, 0.00]   Δy ∈ [-0.15, 0.15]
tray:    Δx ∈ [-0.10, 0.10]   Δy ∈ [-0.25, 0.25]
# object and tray zones do not overlap — 5 cm safety gap along x
```

### 3.2. Observations (concatenated → 32 dim)

| Term | Dim | Description |
|---|---|---|
| `joint_pos` (relative) | 9 | 7 arm + 2 finger |
| `joint_vel` (relative) | 9 | same |
| `screw_position` | 3 | object position in robot root frame |
| `tray_position` | 3 | tray position in robot root frame |
| `last_action` | 8 | 7 arm + 1 gripper (binary) |

`enable_corruption=True` — observations are noised for robustness.

### 3.3. Actions (8 dim)

| Term | Type | Dim | Description |
|---|---|---|---|
| `arm_action` | `JointPositionActionCfg` | 7 | target joint positions, `scale=0.5`, `use_default_offset=True` |
| `gripper_action` | `BinaryJointPositionActionCfg` | 1 | binary: `>0` → close (0.0), `≤0` → open (0.044) |

### 3.4. Sim parameters

```python
decimation = 2              # 1 agent action per 2 sim steps
episode_length_s = 5        # 250 agent steps (500 sim steps)
sim.dt = 0.01               # 100 Hz physics
num_envs = 4096
env_spacing = 4.0

# PhysX GPU buffers (for 4096 envs with contacts)
gpu_max_rigid_patch_count = 338090
gpu_found_lost_aggregate_pairs_capacity = 4 * 1024 * 1024
gpu_total_aggregate_pairs_capacity = 32768
bounce_threshold_velocity = 0.01
friction_correlation_distance = 0.00625
```

---

## 4. Rewards

7 terms in [openarm_pnp_env_cfg.py](source/openarm_pnp/openarm_pnp/tasks/manager_based/openarm_pnp/openarm_pnp_env_cfg.py),
implementation in [mdp/rewards.py](source/openarm_pnp/openarm_pnp/tasks/manager_based/openarm_pnp/mdp/rewards.py).

| Term | Weight | Parameters | Purpose |
|---|---|---|---|
| `reaching_screw` | **+1.0** | `std=0.1` | `tanh`: EE → object, always active |
| `screw_lifted` | **+15.0** | `minimal_height=0.81` | `+15`/step while object is above threshold (constant baseline) |
| `screw_goal_tracking` | **+16.0** | `std=0.3`, `target_z_offset=0.05` | `tanh`: object → point INSIDE the tray, gated by lift |
| `screw_placed` | **+50.0** | `h_thr=0.08`, `vel_thr=0.1` | bonus when object is ≈ at rest inside the tray |
| `success_bonus` | **+200.0** | `term_keys="screw_placed_success"` | one-shot bonus on the successful termination step |
| `action_rate` | **−1e-4** | clamped (−1000, +1000) | action smoothness |
| `joint_vel` | **−1e-4** | clamped (−1000, +1000) | motion smoothness |

Pseudocode (simplified):

```python
reaching_screw       = 1 - tanh(||screw - ee|| / 0.1)
screw_lifted         = 1.0 if screw_z > 0.81 else 0.0
screw_goal_tracking  = lifted * (1 - tanh(||screw - (tray + 0.05z)|| / 0.3))
screw_placed         = (||screw.xy - tray.xy|| < 0.08) & (||screw.vel|| < 0.1)
success_bonus        = 1.0 once when screw_placed_success fires
action_rate          = sum((a_t - a_{t-1})^2).clamp(-1000, 1000)
joint_vel            = sum(joint_vel^2).clamp(-1000, 1000)
```

### Curriculum

After **10 000 steps** the penalty weights are scaled up 1000×:

```text
action_rate: -1e-4 → -1e-1
joint_vel:   -1e-4 → -1e-1
```

### Terminations

| Term | Condition | Effect |
|---|---|---|
| `time_out` | 5 s elapsed (250 steps) | episode ends, no bonus |
| `screw_dropped` | `screw_z < 0.78` (fell off the table) | episode ends, no bonus |
| `screw_placed_success` | object in tray (`xy<0.08`, `0 < Δz < 0.12`, `vel<0.2`) | episode ends **+ `success_bonus=200`** |

---

## 5. PPO (rsl_rl)

Config — [agents/rsl_rl_ppo_cfg.py](source/openarm_pnp/openarm_pnp/tasks/manager_based/openarm_pnp/agents/rsl_rl_ppo_cfg.py).

```python
num_steps_per_env = 24
max_iterations    = 2000
save_interval     = 50
experiment_name   = "openarm_pnp"
empirical_normalization = False

policy:
    init_noise_std     = 1.0
    actor_hidden_dims  = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation         = "elu"

algorithm:
    learning_rate          = 1e-4
    schedule               = "adaptive"   # via desired_kl=0.01
    gamma                  = 0.98
    lam                    = 0.95
    entropy_coef           = 0.006
    clip_param             = 0.2
    value_loss_coef        = 1.0
    use_clipped_value_loss = True
    num_learning_epochs    = 5
    num_mini_batches       = 4
    max_grad_norm          = 1.0
```

> PPO with adaptive KL often **collapses after ~2500 iterations** — that's why
> `max_iterations=2000` and checkpoints are saved frequently. If the policy "turns away
> from the table", roll back to an earlier checkpoint.

---

## 6. Project layout

```
openarm_pnp/
├── README.md                            # this file
├── BASELINE.md                          # snapshot of the working configuration
├── AGENT.md                             # detailed agent documentation
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py, random_agent.py   # env sanity-check
│   └── rsl_rl/{train.py, play.py}
└── source/openarm_pnp/openarm_pnp/
    └── tasks/manager_based/
        ├── assets/openarm_unimanual.py  # robot ArticulationCfg
        └── openarm_pnp/
            ├── openarm_pnp_env_cfg.py   # main environment
            ├── mdp/
            │   ├── rewards.py
            │   ├── terminations.py
            │   └── observations.py
            └── agents/rsl_rl_ppo_cfg.py # PPO hyperparameters
```

---

## 7. Known pitfalls

- **Cube hovers/shakes above the tray** → `velocity_threshold` in `screw_placed` is too lax.
- **Pushes the cube into the tray wall** → `horizontal_threshold` > the tray's inner radius.
- **Grasps empty air** → check `reaching std` and the `minimal_height` for lift.
- **`Policy/mean_noise_std` > 2.0** → collapse. Lower `entropy_coef` or `max_iterations`.
- **`Loss/value_function = inf`** → the `*_clamped` penalty versions are not wired up.
- **Robot "turns away" from the table** → over-training, use an earlier checkpoint.
