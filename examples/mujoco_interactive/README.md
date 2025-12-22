# Mujoco Interactive ALOHA Example

This example launches a visual Mujoco ALOHA simulation and drives it with OpenPI's most generalizable ALOHA policy (`pi0_base`).
It prompts you for language instructions between episodes so you can interactively test action-following behavior.

## 1) Start the policy server

Run the policy server with the base, generalist ALOHA checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha --policy.dir=gs://openpi-assets/checkpoints/pi0_base
```

## 2) Install dependencies

```bash
uv venv --python 3.10 examples/mujoco_interactive/.venv
source examples/mujoco_interactive/.venv/bin/activate
uv pip sync examples/mujoco_interactive/requirements.txt
uv pip install -e packages/openpi-client
```

If you see EGL errors, install the following system dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

## 3) Run the interactive simulator

```bash
MUJOCO_GL=glfw python examples/mujoco_interactive/main.py
```

You can change tasks, prompts, and episode length via flags:

```bash
MUJOCO_GL=glfw python examples/mujoco_interactive/main.py \
  --task gym_aloha/AlohaTransferCube-v0 \
  --prompt "pick up the cube" \
  --max_steps 300
```

At the start of every episode, the script will prompt you to enter a new instruction (or reuse the previous one). Type
`quit` to exit.
