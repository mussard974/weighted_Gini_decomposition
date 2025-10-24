# Decomposition of the weighted $\alpha$-Gini index

**Weighted (α, β)-Gini Decomposition in PyTorch**

This repository provides a modular Python class for computing **multi-level weighted Gini decompositions** following the methods introduced in:

- Mornet, P., Zoli, C., Mussard, S., Sadefo-Kamdem, J., Seyte, F., Terraza, M. (2013).  
  *The (α, β)-multi-level α-Gini decomposition with an illustration to income inequality in France in 2005.*  
  *Economic Modelling*, 35(C), 944–963.

- Mussard, S. & Mornet, P. (2019).  
  *A Note on α-Gini Measures.*  
  *Review of Income and Wealth*, 65: 675–682.  
  DOI: [10.1111/roiw.12373](https://doi.org/10.1111/roiw.12373)

---

## 📦 Features

- Weighted within-group and between-group Gini computation  
- Support for α-Gini and (α, β)-Gini indices  
- PyTorch tensor-based implementation for efficiency  
- Dagum’s decomposition with distances  
- Works directly with pandas DataFrames  
- Summary tables (via `PrettyTable`)

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/GiniDecomposition.git
cd GiniDecomposition
pip install -r requirements.txt
