import dataclasses
import logging
import queue
import threading
import time
from typing import Iterable

import gymnasium
import numpy as np
import tyro
from gymnasium.wrappers import HumanRendering
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Run a MuJoCo simulation driven by a remote OpenPI policy server."""

    env_id: str = "Ant-v4"
    seed: int = 0

    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 15
    control_hz: float = 20.0

    prompt: str = "walk forward"


def _pad_or_trim(values: Iterable[float], size: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size >= size:
        return array[:size]
    padded = np.zeros(size, dtype=np.float32)
    padded[: array.size] = array
    return padded


def _extract_joint_position(env: gymnasium.Env, size: int = 7) -> np.ndarray:
    qpos = None
    if hasattr(env.unwrapped, "data") and hasattr(env.unwrapped.data, "qpos"):
        qpos = np.asarray(env.unwrapped.data.qpos).ravel()
    elif hasattr(env.unwrapped, "state_vector"):
        qpos = np.asarray(env.unwrapped.state_vector()).ravel()
    if qpos is None:
        qpos = np.zeros(size, dtype=np.float32)
    return _pad_or_trim(qpos, size)


def _build_observation(env: gymnasium.Env, image: np.ndarray, prompt: str) -> dict:
    joint_position = _extract_joint_position(env)
    gripper_position = np.zeros(1, dtype=np.float32)

    return {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }


def _match_action_space(action: np.ndarray, action_space: gymnasium.Space) -> np.ndarray:
    if not isinstance(action_space, gymnasium.spaces.Box):
        raise ValueError("This example only supports Box action spaces.")

    target_dim = int(np.prod(action_space.shape))
    action = np.asarray(action, dtype=np.float32).ravel()
    if action.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: action.size] = action
        action = padded
    elif action.size > target_dim:
        action = action[:target_dim]
    action = action.reshape(action_space.shape)
    return np.clip(action, action_space.low, action_space.high)


def _prompt_reader(commands: queue.Queue[str]) -> None:
    while True:
        try:
            line = input("prompt> ").strip()
        except EOFError:
            return
        commands.put(line)
        if line in {":quit", ":exit"}:
            return


def main(args: Args) -> None:
    base_env = gymnasium.make(args.env_id, render_mode="rgb_array")
    env = HumanRendering(base_env)

    policy = action_chunk_broker.ActionChunkBroker(
        policy=websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        ),
        action_horizon=args.action_horizon,
    )

    commands: queue.Queue[str] = queue.Queue()
    prompt_thread = threading.Thread(target=_prompt_reader, args=(commands,), daemon=True)
    prompt_thread.start()

    current_prompt = args.prompt
    last_step = time.time()

    env.reset(seed=args.seed)

    logger.info("Type a new prompt at any time, or :reset / :quit")

    while True:
        while not commands.empty():
            command = commands.get_nowait()
            if command in {":quit", ":exit"}:
                env.close()
                return
            if command == ":reset":
                env.reset()
                policy.reset()
                continue
            if command:
                current_prompt = command
                logger.info("Updated prompt to: %s", current_prompt)

        image = env.render()
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        observation = _build_observation(env, image, current_prompt)

        action_dict = policy.infer(observation)
        action = action_dict["actions"]
        action = _match_action_space(action, env.action_space)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            policy.reset()

        elapsed = time.time() - last_step
        sleep_time = max(0.0, (1.0 / args.control_hz) - elapsed)
        if sleep_time:
            time.sleep(sleep_time)
        last_step = time.time()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
