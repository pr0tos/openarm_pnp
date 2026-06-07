# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--homing-steps",
    type=int,
    default=60,
    help="After a successful place, override the policy with a 'go to default joint_pos' command for this many "
    "agent steps before the env resets. Set to 0 to disable.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm_pnp.tasks  # noqa: F401

from isaaclab.managers import SceneEntityCfg
from openarm_pnp.tasks.manager_based.openarm_pnp.mdp.terminations import screw_in_tray


class HomeAfterSuccess:
    """Drives the robot back to its default joint pose after a successful place.

    When `screw_in_tray` first becomes true for an env, this controller latches a per-env
    countdown of `homing_steps` agent steps. While the countdown is active it overrides the
    policy output: zero arm action (interpreted as `target = default_joint_pos` because
    `arm_action.use_default_offset=True`) and a negative gripper action (open). When the
    countdown reaches zero the policy regains control until `time_out` resets the env.

    Requires that the `screw_placed_success` termination is disabled, otherwise the env
    auto-resets on success and there is no time to play the homing motion.
    """

    # success-check thresholds — mirror the disabled termination
    _XY_THRESHOLD = 0.08
    _MAX_HEIGHT_ABOVE_TRAY = 0.12
    _MAX_VEL = 0.2

    def __init__(self, env, homing_steps: int):
        self._env = env.unwrapped
        self._homing_steps = int(homing_steps)
        self._counter = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self._env.device)
        self._screw_cfg = SceneEntityCfg("screw")
        self._tray_cfg = SceneEntityCfg("tray")

    def _is_active(self) -> torch.Tensor:
        return self._counter > 0

    def override(self, actions: torch.Tensor) -> torch.Tensor:
        """Detect success and overwrite actions for envs currently in the homing phase."""
        success = screw_in_tray(
            self._env,
            screw_cfg=self._screw_cfg,
            tray_cfg=self._tray_cfg,
            xy_threshold=self._XY_THRESHOLD,
            max_height_above_tray=self._MAX_HEIGHT_ABOVE_TRAY,
            max_vel=self._MAX_VEL,
        )
        # latch the countdown on the rising edge of success
        rising_edge = success & ~self._is_active()
        self._counter = torch.where(
            rising_edge,
            torch.full_like(self._counter, self._homing_steps),
            self._counter,
        )

        active = self._is_active()
        if active.any():
            # zero arm action ⇒ target = default joint_pos (use_default_offset=True)
            actions[active, :7] = 0.0
            # binary gripper: negative ⇒ open
            actions[active, 7] = -1.0
            self._counter = (self._counter - 1).clamp_(min=0)

        return actions

    def on_reset(self, dones: torch.Tensor) -> None:
        """Clear the countdown for envs that have just been reset."""
        if dones.any():
            self._counter = torch.where(dones.bool(), torch.zeros_like(self._counter), self._counter)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable the success termination so the episode does not auto-reset on placement —
    # we want time for HomeAfterSuccess to drive the robot back to its default pose first.
    # The episode will still end on time_out (or screw_dropped). The success_bonus reward
    # references this termination via term_keys, so it has to go too (rewards are irrelevant
    # during play anyway).
    if args_cli.homing_steps > 0 and getattr(env_cfg.terminations, "screw_placed_success", None) is not None:
        env_cfg.terminations.screw_placed_success = None
        if getattr(env_cfg.rewards, "success_bonus", None) is not None:
            env_cfg.rewards.success_bonus = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # homing controller: drives the arm to its default pose after a successful place
    home_after_success = HomeAfterSuccess(env, homing_steps=args_cli.homing_steps) if args_cli.homing_steps > 0 else None

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if home_after_success is not None:
                actions = home_after_success.override(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if home_after_success is not None:
                home_after_success.on_reset(dones)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
