---
name: Bug Report
about: Report a bug or unexpected behaviour
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the Bug

A clear and concise description of what the bug is.

## Minimal Reproducing Example

```python
from siton.sdk import *

# Paste the smallest strategy/code that triggers the bug
STRATEGY = Strategy(
    name="...",
    signal=ema_cross(fast=[8], slow=[21]),
)
```

```bash
# Command used
siton my_strategy.py --demo
```

## Expected Behaviour

What did you expect to happen?

## Actual Behaviour

What actually happened? Include the full traceback if applicable.

```
Traceback (most recent call last):
  ...
```

## Environment

| Field | Value |
|---|---|
| Siton version | (run `pip show siton`) |
| Python version | (run `python --version`) |
| OS | (e.g. Ubuntu 22.04 / macOS 14 / Windows 11) |
| NumPy version | (run `python -c "import numpy; print(numpy.__version__)"`) |
| Numba version | (run `python -c "import numba; print(numba.__version__)"`) |

## Additional Context

Any other context about the problem here.
