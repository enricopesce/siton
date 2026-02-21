---
name: Feature Request
about: Suggest a new indicator, SDK feature, or engine improvement
title: "[Feature] "
labels: enhancement
assignees: ''
---

## Summary

A clear one-sentence description of the feature.

## Motivation

What problem does this solve? What use-case does it enable?

## Proposed API (if applicable)

Show what the feature would look like from the user's perspective.

```python
# Example: new indicator
sig = stochastic(k_period=[14], d_period=[3], oversold=[20], overbought=[80])

# Example: new composition operator
entry = trend.hold_for(n=3)  # hold signal active for N bars after trigger
```

## Alternatives Considered

Have you tried working around this with the existing API? What was missing?

## Additional Context

Links to papers, other frameworks, screenshots, or any other relevant context.
