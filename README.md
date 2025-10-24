# Decomposition of the weighted $\alpha$-Gini index in *Torch*

---

This repository provides a modular Python class for computing the **weighted $\alpha$-Gini index decompositions** 

**Sampling weights** are introduced in the following decompositions: 

- Mornet, P., Zoli, C., Mussard, S., Sadefo-Kamdem, J., Seyte, F., Terraza, M. (2013).  
  *The (α, β)-multi-level α-Gini decomposition with an illustration to income inequality in France in 2005.*  
  *Economic Modelling*, 35(C), 944–963.
  DOI: 10.1016/j.econmod.2013.06.016 

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
