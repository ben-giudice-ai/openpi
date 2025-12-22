import dataclasses
import logging
import time
from typing import Iterable

import gymnasium as gym
import numpy as np
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
import tyro

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for visualization
    cv2 = None

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    env_id: str = "Ant-v4"
    seed: int = 0
    episode_horizon: int = 400
    prompt: str = "walk forward"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_droid"
    config_name: str = "pi05_droid"
    max_fps: float = 20.0
    render_window_name: str = "openpi mujoco"


def _render_image(env: gym.Env, camera_candidates: Iterable[str]) -> np.ndarray:
    for camera_name in camera_candidates:
        try:
            return env.render(camera_name=camera_name)
        except Exception:
            continue
    return env.render()


def _build_droid_observation(obs: np.ndarray, image: np.ndarray, prompt: str) -> dict:
    joint_state = np.zeros(7, dtype=np.float32)
    flattened = np.asarray(obs, dtype=np.float32).ravel()
    joint_state[: min(len(flattened), 7)] = flattened[:7]

    return {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
        "observation/joint_position": joint_state,
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
    }


def _maybe_show_frame(window_name: str, frame: np.ndarray) -> int | None:
    if cv2 is None:
        return None
    cv2.imshow(window_name, frame[:, :, ::-1])
    return cv2.waitKey(1) & 0xFF


def _sleep_for_fps(max_fps: float, last_step: float) -> float:
    if max_fps <= 0:
        return time.time()
    target_dt = 1.0 / max_fps
    elapsed = time.time() - last_step
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)
    return time.time()


def main(args: Args) -> None:
    env = gym.make(args.env_id, render_mode="rgb_array")

    train_config = _config.get_config(args.config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

    prompt = args.prompt
    last_step = time.time()

    while True:
        obs, _ = env.reset(seed=args.seed)
        logger.info("Current prompt: %s", prompt)
        logger.info("Press 'n' to enter a new prompt, 'q' to quit.")

        for _ in range(args.episode_horizon):
            frame = _render_image(env, ["track", "front", "side", "top"])
            droid_obs = _build_droid_observation(obs, frame, prompt)
            action_chunk = policy.infer(droid_obs)["actions"]
            action = np.asarray(action_chunk[0], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, _, terminated, truncated, _ = env.step(action)

            key = _maybe_show_frame(args.render_window_name, frame)
            if key == ord("q"):
                env.close()
                return
            if key == ord("n"):
                break

            if terminated or truncated:
                break

            last_step = _sleep_for_fps(args.max_fps, last_step)

        if cv2 is not None:
            cv2.destroyAllWindows()

        new_prompt = input("Enter a new prompt (or press enter to reuse): ").strip()
        if new_prompt:
            prompt = new_prompt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
