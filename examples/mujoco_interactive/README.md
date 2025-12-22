# MuJoCo interactive example

This example runs a MuJoCo simulation with live rendering and uses openpi's most generalist policy
(pi05-DROID) to generate actions from a language prompt. You can swap prompts interactively while the
sim is running.

## Setup

```bash
uv pip install gymnasium[mujoco] opencv-python
```

## Run

```bash
uv run examples/mujoco_interactive/main.py \
  --env-id Ant-v4 \
  --prompt "walk forward"
```

## Controls

- **n**: stop the current rollout and enter a new prompt.
- **q**: quit the script.

If you don't have OpenCV installed, the policy will still run, but you won't see a render window.

## Customizing

- Change `--checkpoint` or `--config-name` to point to another openpi policy.
- Use `--episode-horizon` to control how many steps are generated per prompt.
- Use `--max-fps` to throttle rendering.
