import dataclasses
import logging
import time

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

import env as _env


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    task: str = "gym_aloha/AlohaTransferCube-v0"
    obs_type: str = "pixels_agent_pos"
    seed: int = 0

    action_horizon: int = 10
    max_steps: int = 300
    max_hz: float = 50.0

    host: str = "0.0.0.0"
    port: int = 8000

    render: bool = True
    prompt: str = "pick up the cube"
    num_episodes: int = 0


def _prompt_for_instruction(current_prompt: str) -> str | None:
    while True:
        user_input = input(
            "Enter instruction (blank to reuse current, 'quit' to exit): "
        ).strip()
        if user_input.lower() in {"quit", "exit"}:
            return None
        if user_input:
            return user_input
        if current_prompt:
            return current_prompt
        print("Please provide a non-empty instruction.")


def run_episode(
    env: _env.MujocoAlohaInteractiveEnvironment,
    policy: action_chunk_broker.ActionChunkBroker,
    *,
    prompt: str,
    max_steps: int,
    max_hz: float,
) -> int:
    env.set_prompt(prompt)
    env.reset()
    policy.reset()

    step_time = 1 / max_hz if max_hz > 0 else 0
    last_step_time = time.time()
    steps = 0

    while not env.is_episode_complete() and (max_steps <= 0 or steps < max_steps):
        observation = env.get_observation()
        action = policy.infer(observation)
        env.apply_action(action)
        steps += 1

        if step_time > 0:
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

    return steps


def main(args: Args) -> None:
    env = _env.MujocoAlohaInteractiveEnvironment(
        task=args.task,
        obs_type=args.obs_type,
        seed=args.seed,
        render_mode="human" if args.render else None,
    )

    policy = action_chunk_broker.ActionChunkBroker(
        policy=_websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        ),
        action_horizon=args.action_horizon,
    )

    episode = 0
    prompt = args.prompt

    while args.num_episodes <= 0 or episode < args.num_episodes:
        prompt = _prompt_for_instruction(prompt)
        if prompt is None:
            logging.info("Exiting on user request.")
            break

        steps = run_episode(
            env,
            policy,
            prompt=prompt,
            max_steps=args.max_steps,
            max_hz=args.max_hz,
        )
        logging.info(
            "Episode %d complete after %d steps (reward=%.3f).",
            episode + 1,
            steps,
            env.episode_reward,
        )
        episode += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
