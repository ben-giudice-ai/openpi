import dataclasses
import logging
import pathlib
import sys
import threading


class CommandState:
    def __init__(self, default_prompt: str | None) -> None:
        self._default_prompt = default_prompt
        self._command = ""
        self._lock = threading.Lock()

    def update(self, command: str) -> None:
        with self._lock:
            self._command = command

    def get(self) -> str | None:
        with self._lock:
            if self._command:
                return self._command
        return self._default_prompt


def _stdin_command_loop(command_state: CommandState) -> None:
    for line in sys.stdin:
        command = line.strip()
        if not command:
            continue
        command_state.update(command)
        logging.info("Updated command: %s", command)


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim_realtime/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    default_prompt: str | None = "Pick up the cube and place it in the bowl."


def main(args: Args) -> None:
    import env as _env
    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy as _websocket_client_policy
    from openpi_client.runtime import environment as _environment
    from openpi_client.runtime import runtime as _runtime
    from openpi_client.runtime.agents import policy_agent as _policy_agent
    import saver as _saver

    class PromptedEnvironment(_environment.Environment):
        def __init__(self, environment: _environment.Environment, command_state: CommandState) -> None:
            self._environment = environment
            self._command_state = command_state

        def reset(self) -> None:
            self._environment.reset()

        def is_episode_complete(self) -> bool:
            return self._environment.is_episode_complete()

        def get_observation(self) -> dict:
            observation = dict(self._environment.get_observation())
            prompt = self._command_state.get()
            if prompt:
                observation["prompt"] = prompt
            return observation

        def apply_action(self, action: dict) -> None:
            self._environment.apply_action(action)

    command_state = CommandState(default_prompt=args.default_prompt)
    stdin_thread = threading.Thread(target=_stdin_command_loop, args=(command_state,), daemon=True)
    stdin_thread.start()

    environment = PromptedEnvironment(
        _env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        command_state,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    import tyro

    tyro.cli(main)
