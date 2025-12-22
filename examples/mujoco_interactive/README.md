# MuJoCo Interactive Demo (Pi05-DROID)

This example spins up a minimal MuJoCo scene and streams observations to the **π₀.₅-DROID** policy (our most generalizable
checkpoint). You can update the language prompt at runtime to steer the policy's actions.

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/mujoco_interactive/.venv
source examples/mujoco_interactive/.venv/bin/activate
uv pip sync examples/mujoco_interactive/requirements.txt
uv pip install -e packages/openpi-client

# Run the MuJoCo demo
python examples/mujoco_interactive/main.py
```

Terminal window 2:

```bash
# Run the policy server with the most generalizable checkpoint
uv run scripts/serve_policy.py --env DROID
```

## Interactive prompting

While the simulation is running, type a new prompt into the terminal and press Enter, e.g.:

```
move the ball to the top right
circle around the origin
```

The next action chunk will follow the updated prompt.
