# Run Aloha Sim (Realtime Commands)

This example mirrors `examples/aloha_sim` but keeps a live prompt that you can update while the
simulation runs.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM --default-prompt 'Pick up the cube and place it in the bowl.'"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim_realtime/.venv
source examples/aloha_sim_realtime/.venv/bin/activate
uv pip sync examples/aloha_sim_realtime/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation (MuJoCo rendering)
MUJOCO_GL=egl python examples/aloha_sim_realtime/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM --default-prompt "Pick up the cube and place it in the bowl."
```

## Updating commands live

While `main.py` is running, type a new command into the terminal and press Enter. The latest
command is injected into each observation as `prompt` before being sent to the policy.
If you have not typed anything yet, the `--default-prompt` (or `Args.default_prompt`) is used.
