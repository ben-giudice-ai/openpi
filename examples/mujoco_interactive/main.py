import collections
import dataclasses
import logging
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


_MODEL_XML = """
<mujoco model="openpi_point_mass">
  <option timestep="0.02"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
    <body name="agent" pos="0 0 0.05">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-1 1" damping="1"/>
      <joint name="slide_y" type="slide" axis="0 1 0" range="-1 1" damping="1"/>
      <geom type="sphere" size="0.05" rgba="0.1 0.6 0.9 1"/>
    </body>
    <camera name="main" pos="0 -1.6 1.0" xyaxes="1 0 0 0 0.8 0.6"/>
  </worldbody>
  <actuator>
    <motor joint="slide_x" ctrlrange="-1 1" gear="1"/>
    <motor joint="slide_y" ctrlrange="-1 1" gear="1"/>
  </actuator>
</mujoco>
"""


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "move the blue ball around"
    resize_size: int = 224
    replan_steps: int = 5
    render_width: int = 512
    render_height: int = 512
    max_steps: int = 2_000


class PromptState:
    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def get(self) -> str:
        with self._lock:
            return self._prompt

    def set(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt


def _prompt_listener(state: PromptState) -> None:
    while not state.stop_event.is_set():
        try:
            new_prompt = input("New prompt (press Enter to keep current): ").strip()
        except EOFError:
            break
        if new_prompt:
            state.set(new_prompt)
            logging.info("Updated prompt to: %s", new_prompt)


def _render_observation(renderer: mujoco.Renderer, data: mujoco.MjData, *, resize_size: int) -> np.ndarray:
    renderer.update_scene(data, camera="main")
    image = renderer.render()
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    return image


def _build_policy_input(image: np.ndarray, qpos: np.ndarray, prompt: str) -> dict:
    joint_position = np.zeros(7, dtype=np.float32)
    count = min(7, qpos.shape[0])
    joint_position[:count] = qpos[:count]
    return {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
        "observation/joint_position": joint_position,
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
    }


def _map_action(action: np.ndarray, num_actuators: int) -> np.ndarray:
    if action.shape[0] < num_actuators:
        padded = np.zeros(num_actuators, dtype=np.float32)
        padded[: action.shape[0]] = action
        action = padded
    return np.clip(action[:num_actuators], -1.0, 1.0)


def main(args: Args) -> None:
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    action_plan = collections.deque()

    prompt_state = PromptState(args.prompt)
    listener = threading.Thread(target=_prompt_listener, args=(prompt_state,), daemon=True)
    listener.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        steps = 0
        while viewer.is_running() and steps < args.max_steps:
            if not action_plan:
                image = _render_observation(renderer, data, resize_size=args.resize_size)
                policy_input = _build_policy_input(image, data.qpos.copy(), prompt_state.get())
                action_chunk = client.infer(policy_input)["actions"]
                action_plan.extend(action_chunk[: args.replan_steps])

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            data.ctrl[:] = _map_action(action, model.nu)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
            steps += 1

    prompt_state.stop_event.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
