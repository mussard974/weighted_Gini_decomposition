# Decomposition of the weighted $\alpha$-Gini index in *Torch*

---

This repository provides a modular Python class for computing the **weighted $\alpha$-Gini index decompositions** 

**Sampling weights** are introduced in the following decompositions: 

- Dagum, C. *A new approach to the decomposition of the Gini income inequality ratio*.
  *Empirical Economics* 22, 515–531 (1997).
  DOI: [doi.org/10.1007/BF01205777](https://link.springer.com/article/10.1007/BF01205777)

- Mornet, P., Zoli, C., Mussard, S., Sadefo-Kamdem, J., Seyte, F., Terraza, M. (2013).  
  *The (α, β)-multi-level α-Gini decomposition with an illustration to income inequality in France in 2005.*  
  *Economic Modelling*, 35(C), 944–963.
  DOI: [10.1016/j.econmod.2013.06.016](https://www.sciencedirect.com/science/article/abs/pii/S0264999313002332) 

- Mussard, S. & Mornet, P. (2019).  
  *A Note on α-Gini Measures.*  
  *Review of Income and Wealth*, 65: 675–682.  
  DOI: [10.1111/roiw.12373](https://doi.org/10.1111/roiw.12373)

---

## Functions

- Weighted within-group and between-group Gini 
- α-Gini and (α, β)-Gini decompositions  
- PyTorch tensor-based implementation for efficiency  
- Dagum’s decomposition with distances (β-distance)
- Works directly with pandas DataFrames  
- Summary tables (via `PrettyTable`)

---

## Requirements

pandas
torch
prettytable

---

## Example

```python
import pandas as pd
from Gini_decomp_weight import GiniDecomposition

# Example dataset
df = pd.DataFrame({
    "income": [2000, 2500, 3000, 3500, 4000, 4500],
    "group": ["A", "A", "B", "B", "C", "C"],
    "weight": [1.2, 0.8, 1.0, 1.0, 0.9, 1.1]
})
```

**Instantiate and fit the decomposition**
Note that if $\alpha=2$ this corresponds to the weighted coefficient of variation squared

```python
model = GiniDecomposition(alpha=2)
model.fit(data=df, value="income", group="group", weight="weight")
# Show summary
model.summary()
```

**Weighted Gini decomposition: 2 components**

| Component                        | Index   |
|----------------------------------|----------|
| Gini within groups (`G_w`)       | 0.0019   |
| Gini between groups (`G_gb`)     | 0.0722   |
| Gini total (`G_w + G_gb`)        | 0.0741   |
| Gini total (verification)        | 0.0741   |

**Gini decomposition: 3 components**

| Component                        | Index   |
|----------------------------------|----------|
| Gini within groups (`G_w`)       | 0.0019   |
| Gini net between groups (`G_nb`) | 0.0722   |
| Gini transvariation (`G_t`)      | 0.0000   |
| Gini total (`G_w + G_nb + G_t`)  | 0.0741   |



**Fit Dagum's Gini decomposition with weights: $\alpha=1$**

* Note that if $\alpha=1$ this corresponds to the standard weighted Gini index

```python
model = GiniDecomposition(alpha=1)
model.fit(data=df, value="income", group="group", weight="weight")
# Show summary
gini.summary()
```

**Fit the absolute weighted $\alpha$-Gini index**

* The weighted absolute $\alpha$-Gini index:
$$
\mathcal G_\alpha = 
\frac{
\displaystyle \sum_i \sum_j w_i w_j \, |x_i - x_j|^{\alpha}
}{
2\,(\sum_iw_i)^2
$$


```python
model = GiniDecomposition(alpha=2, method='absolute')
model.fit(data=df, value="income", group="group", weight="weight")
# Show summary
gini.summary()
```

**Fit the $(\alpha-\beta)$-decomposition of the Gini index**
```python
model = GiniDecomposition(alpha=2, beta = 2)
model.fit(data=df, value="income", group="group", weight="weight")
# Show summary
gini.summary()
```
