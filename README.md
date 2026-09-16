# Numerical Analysis

## Implicit Euler method

`implicit_euler.py` implements backward (implicit) Euler for scalar initial
value problems, including a Newton solve at every step.  It accepts an optional
analytic Jacobian; otherwise it estimates `df/dy` with centered finite
differences.

```python
from implicit_euler import implicit_euler

solution = implicit_euler(
    lambda _t, y: -15.0 * y,
    t0=0.0,
    y0=1.0,
    step_size=0.1,
    steps=10,
    jacobian=lambda _t, _y: -15.0,
)
print(solution.values[-1])
```

Run the included example with `python implicit_euler.py`, or run the tests with
`python -m unittest -v`.
