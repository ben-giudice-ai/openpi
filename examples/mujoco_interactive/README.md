# MuJoCo Interactive Action Following

This example streams observations from a MuJoCo simulation to an OpenPI policy server and
applies the returned actions in real time. It opens a live MuJoCo viewer and lets you update
the natural-language prompt on the fly.

## 1) Start the policy server (recommended: π0.5 DROID)

The π0.5 DROID checkpoint is OpenPI's most generalisable model. Start a server with it in a
separate terminal:

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

## 2) Install dependencies

```bash
uv pip install -r examples/mujoco_interactive/requirements.txt
```

## 3) Run the MuJoCo simulation

```bash
uv run examples/mujoco_interactive/main.py \
  --env-id Ant-v4 \
  --prompt "walk forward"
```

### Interactive controls

While the simulation is running, type into the terminal:

- Any text updates the prompt sent to the policy server.
- `:reset` resets the environment and policy state.
- `:quit` exits the loop.

The example maps the policy output to the environment action space (trimming or padding as
needed) and clips to the environment bounds.
