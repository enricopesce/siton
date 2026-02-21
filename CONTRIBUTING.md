# Contributing to Siton

Thank you for taking the time to contribute! All contributions are welcome — bug reports, feature requests, documentation fixes, new indicators, and code improvements.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Development Setup](#development-setup)
- [Adding a New Indicator](#adding-a-new-indicator)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/enricopesce/siton.git
   cd siton
   ```
3. Set up your dev environment (see [Development Setup](#development-setup)).
4. Create a branch for your change:
   ```bash
   git checkout -b feat/my-feature
   # or
   git checkout -b fix/some-bug
   ```

---

## How to Contribute

### Reporting Bugs

Please use the **Bug Report** issue template on GitHub. Include:

- Siton version (`pip show siton`)
- Python version (`python --version`)
- Minimal reproducing code snippet
- The full traceback / error message
- What you expected vs. what actually happened

### Suggesting Features

Use the **Feature Request** issue template. Describe the use-case clearly — what problem does the feature solve, and what would the API look like?

### Submitting a Pull Request

1. Make sure your branch is up to date with `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. Keep your PR focused — one feature or fix per PR.
3. Run a quick smoke test before pushing:
   ```bash
   siton examples/01_ema_crossover.py --demo
   siton examples/03_trend_momentum_confluence.py --demo
   ```
4. Fill in the PR template.
5. A maintainer will review your PR. Please respond to review comments promptly.

---

## Development Setup

### Prerequisites

The TA-Lib C library must be installed before `pip install`:

```bash
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib
```

### Install in editable mode

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs Siton in editable mode plus the dev dependencies (`pytest`, `ruff`, `mypy`).

### Running checks

```bash
# Lint
ruff check siton/

# Format
ruff format siton/

# Type check
mypy siton/

# Tests (once the test suite exists)
pytest
```

---

## Adding a New Indicator

All indicator constructors live in `siton/sdk.py`. Follow these steps:

1. **Find a similar indicator** in `sdk.py` to use as a template (e.g. `ema_cross` for MA crossovers, `rsi` for oscillators).

2. **Write the factory**:
   ```python
   def my_indicator(
       periods: list[int] = [14],
       threshold: list[float] = [0.5],
   ) -> Signal:
       """One-line description of what this signal does.

       Returns +1 (long) when ..., -1 (short) when ..., 0 (flat) otherwise.
       """
       def _factory():
           name = "my_indicator"
           grid = {"period": periods, "threshold": threshold}

           def _signal(close, high, low, volume, period, threshold):
               result = _cached(talib.MY_FUNC, "my_indicator", close, timeperiod=period)
               sig = np.zeros(len(close))
               sig[result > threshold] = 1.0
               sig[result < -threshold] = -1.0
               return sig

           return name, grid, _signal

       return Signal(_labeled_factory(_factory, "my_indicator"))
   ```

3. **Add it to the module exports** — update the `__all__` list if one exists, or simply verify `from siton.sdk import *` exposes it.

4. **Add an entry to the Indicator Reference table** in `README.md`.

5. **Smoke-test** it:
   ```python
   from siton.sdk import *
   sig = my_indicator(periods=[10, 20])
   s = Strategy(name="test", signal=sig)
   backtest(s)   # with --demo
   ```

---

## Code Style

- Format with `ruff format` (line length 100).
- Lint with `ruff check`.
- No type annotations required in indicator constructors (they are intentionally minimal), but new classes or utilities should be typed.
- Keep indicator docstrings to one line describing what the signal means.

---

## Commit Messages

Use the conventional commits style:

```
feat: add stochastic oscillator indicator
fix: correct ATR stop-loss direction for short positions
docs: add walk-forward example to README
refactor: extract _merge_flat into utils module
test: add unit test for grid expansion
```

---

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
