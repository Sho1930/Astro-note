---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Julia 1.7.1
  language: julia
  name: julia-fast
---

## Test MyST

```{note}
Just a title note

```python
import numpy as np
```
```{tip}
tips!!
```

This is an example of a
math block

$$
z=\sqrt{x^2+y^2}
$$

```{code-cell} ipython3
from myst_nb import glue
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'b-', linewidth=2)

glue("glued_fig", fig, display=False)
```

  
```{math}
\begin{split}
   u'(t) &= f(t,u(t)), \qquad a \le t \le b,  \\
  u(a) &=u_0.
\end{split}
```

```{code-cell}
:tags: [hide-input]
t = range(0,3,length=800)
u = @. exp(t)*1
lower,upper = @. exp(t)*0.7, @. exp(t)*1.3
plot(t,u,l=:black,ribbon=(lower,upper),
    leg=:none,xlabel=L"t",ylabel=L"u(t)",
    title="Exponential divergence of solutions")
```

## Exercises

1. ✍ For each IVP, determine whether the problem satisfies the conditions of {numref}`Theorem %s <theorem-depIC>`). If so, determine the smallest possible value for $L$.

    **(a)** $f(t,u) = 3 u,\; 0 \le t \le 1$

    **(b)** $f(t,u) = -t \sin(u),\; 0 \le t \le 5$

    **(c)** $f(t,u) = -(1+t^2) u^2,\; 1 \le t \le 3$

    **(d)** $f(t,u) = \sqrt{u},\; 0 \le t \le 1$
