import pandas as pd
import torch
from prettytable import PrettyTable
pd.options.display.float_format = '{:.4f}'.format

class GiniDecomposition(object):
    
    '''
    Weighted Gini (alpha,beta) decomposition

    Adaptation of weights in the following methods:
    ------------
    - Mornet P., Zoli C., Mussard S., Sadefo-Kamdem J., Seyte F., Terraza M. (2013),
      The (α, β)-multi-level α-Gini decomposition with an illustration to income inequality in France in 2005.
      Economic Modelling, vol. 35 (C), pp. 944-963.
    - Mussard, S. and Mornet, P. (2019), A Note on α-Gini Measures.
      Review of Income and Wealth, 65: 675-682. https://doi.org/10.1111/roiw.12373
    '''
    
    def __init__(self, alpha, beta=None, method=None):
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.weight_column = None
    
    # Within-group Gini index with weights    
    def gini_intra(self, x, w=None):
        if w is None:
            w = torch.ones_like(x)
        w = w / w.sum()
        sorted_idx = torch.argsort(x)
        x, w = x[sorted_idx], w[sorted_idx]
        mu = torch.sum(w * x)
        diff = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
        weight_matrix = w.unsqueeze(0) * w.unsqueeze(1)
        G = torch.sum(weight_matrix * diff**self.alpha) / (2 * mu**self.alpha)
        return G
    
    # Between-group Gini index with weights 
    def gini_inter(self, x, y, wx=None, wy=None):
        if wx is None:
            wx = torch.ones_like(x)
        if wy is None:
            wy = torch.ones_like(y)
        wx = wx / wx.sum()
        wy = wy / wy.sum()
        mean_x = torch.sum(wx * x)
        mean_y = torch.sum(wy * y)
        # Pairwise differences
        diff = x.unsqueeze(0) - y.unsqueeze(1)
        abs_diff = torch.abs(diff)
        # Weight matrix
        wmat = wx.unsqueeze(0) * wy.unsqueeze(1)
        # Weighted between-group Gini
        Gini = torch.sum(wmat * abs_diff**self.alpha) / (mean_x**self.alpha + mean_y**self.alpha)
        if mean_x >= mean_y:
            pos = torch.clamp(diff, min=0)   
            neg = torch.clamp(-diff, min=0)  
        else:
            pos = torch.clamp(-diff, min=0)  
            neg = torch.clamp(diff, min=0)   
        # Dagum’s Distance
        num = torch.sum(wmat * (pos**self.alpha)) - torch.sum(wmat * (neg**self.alpha))
        den = torch.sum(wmat * (pos**self.alpha)) + torch.sum(wmat * (neg**self.alpha))
        Distance = num / den
        if self.beta:
            if mean_x >= mean_y:
                pos = torch.clamp(diff, min=0)
                neg = torch.clamp(-diff, min=0)
            else:
                pos = torch.clamp(-diff, min=0)
                neg = torch.clamp(diff, min=0)
            num = torch.sum(wmat * (pos**self.beta)) - torch.sum(wmat * (neg**self.beta))
            den = torch.sum(wmat * (pos**self.beta)) + torch.sum(wmat * (neg**self.beta))
            Distance = num / den
        return Gini, Distance

    def tensor_from_pandas(self, dataframe, values_column, group_column, weight_column=None):
        dataframe[group_column] = dataframe[group_column].astype(str)
        groups = dataframe[group_column].unique()
        x_tensors, w_tensors = [], []
        means, sizes, total_mean = [], [], []
        if weight_column is None or weight_column not in dataframe.columns:
            dataframe['_weight_'] = 1.0
            weight_column = '_weight_'
        else:
            dataframe[weight_column] = dataframe[weight_column].astype(float)
        total_weight = dataframe[weight_column].sum()
        x_all = torch.tensor(dataframe[values_column].astype(float).values, dtype=torch.float64)
        w_all = torch.tensor(dataframe[weight_column].astype(float).values, dtype=torch.float64)
        W = w_all.sum()
        w_all = w_all / w_all.sum()
        mu = torch.sum(w_all * x_all)
        global_mean = W * (mu ** self.alpha) 
        
        for group in groups:
            subset = dataframe[dataframe[group_column] == group]
            x = torch.tensor(subset[values_column].astype(float).values, dtype=torch.float64)
            w = torch.tensor(subset[weight_column].astype(float).values, dtype=torch.float64)
            x_tensors.append(x)
            w_tensors.append(w)

            mean_g = (subset[values_column] * subset[weight_column]).sum() / subset[weight_column].sum()
            size_g = subset[weight_column].sum() / total_weight
            mean_share = (subset[weight_column].sum() * mean_g**self.alpha) / global_mean

            sizes.append(size_g)
            means.append(mean_share)
            total_mean.append(size_g * mean_g)

        means_tensor = torch.tensor(means, dtype=torch.float64)
        size_tensor = torch.tensor(sizes, dtype=torch.float64)
        self.total_mean = sum(total_mean)

        if '_weight_' in dataframe.columns:
            dataframe.drop(columns=['_weight_'], inplace=True, errors='ignore')
        return x_tensors, w_tensors, means_tensor, size_tensor

    def fit(self, data = None, value = None, group = None, weight = None):
        self.dataframe = data
        self.value_column = value
        self.group_column = group
        self.weight_column = weight

        x_tensors, w_tensors, means_tensor, size_tensor = self.tensor_from_pandas(
            data, value, group, weight
        )

        self.num_tensors = len(x_tensors)
        self.means_tensor = means_tensor.clone().detach()
        self.size_tensor = size_tensor.clone().detach()

        # Weighted within-group Ginis
        self.tensor_Gjj = torch.stack([self.gini_intra(x_tensors[i], w_tensors[i])
                                       for i in range(self.num_tensors)])

        # Weighted between-group matrices
        self.matrix_Gij = torch.zeros((self.num_tensors, self.num_tensors), dtype=float)
        self.matrix_Dij = torch.zeros((self.num_tensors, self.num_tensors), dtype=float)

        for i in range(self.num_tensors - 1):
            for j in range(i + 1, self.num_tensors):
                Gij, Dij = self.gini_inter(x_tensors[i], x_tensors[j],
                                           w_tensors[i], w_tensors[j])
                self.matrix_Gij[i, j] = self.matrix_Gij[j, i] = Gij
                self.matrix_Dij[i, j] = self.matrix_Dij[j, i] = Dij

        # Decomposition components
        self.Gini_between = self.means_tensor @ self.matrix_Gij @ self.size_tensor
        self.Gini_net_between = self.means_tensor @ (self.matrix_Gij * self.matrix_Dij) @ self.size_tensor
        self.Gini_transvariation = self.means_tensor @ (self.matrix_Gij * (1 - self.matrix_Dij)) @ self.size_tensor
        self.Gini_within = torch.sum(self.means_tensor * self.tensor_Gjj * self.size_tensor)
        self.Gini_total = self.Gini_between + self.Gini_within

        if self.method == "absolute":
            factor = 2 * self.total_mean**self.alpha
            self.Gini_within *= factor
            self.Gini_net_between *= factor
            self.Gini_between *= factor
            self.Gini_transvariation *= factor
            self.Gini_total *= factor
        
        # Check G total
        x_all = torch.tensor(data[value].astype(float).values, dtype=torch.float64)
        if weight is not None and weight in data.columns:
            w_all = torch.tensor(data[weight].astype(float).values, dtype=torch.float64)
        else:
            w_all = torch.ones_like(x_all)
        w_all = w_all / w_all.sum()
        mu = torch.sum(w_all * x_all)
        diff = torch.abs(x_all.unsqueeze(0) - x_all.unsqueeze(1))
        w_matrix = w_all.unsqueeze(0) * w_all.unsqueeze(1)
        self.Gini_total_verif = torch.sum(w_matrix * diff**self.alpha) / (2 * mu**self.alpha)

    # correlation ratio
    def correlation_ratio(self):
        return self.Gini_between.item() / self.Gini_total.item()

    def summary(self):
        # Table with Gw and Ggb        
        Gw = f"{self.Gini_within.item():.4f}"
        Ggb = f"{self.Gini_between.item():.4f}"
        G = self.Gini_within + self.Gini_between
        table = PrettyTable()
        table.field_names = ["Gini decomposition: 2 components", "  Index  "]
        table.add_row(["Gini within groups (Gw)", Gw])
        table.add_row(["    Gini between groups (Ggb)    ", Ggb])
        table.add_row(["Gini total (Gw + Ggb)", f"{G.item():.4f}"])
        table.add_row(["Gini total (verif.)", f"{self.Gini_total_verif.item():.4f}"])
        print(table)

        # Table with Gw, Gnb and Gt
        Gnb = f"{self.Gini_net_between.item():.4f}"
        Gt = f"{self.Gini_transvariation.item():.4f}"
        G = self.Gini_within + self.Gini_net_between + self.Gini_transvariation
        table = PrettyTable()
        table.field_names = ["Gini decomposition: 3 components", "  Index  "]
        table.add_row(["Gini within groups (Gw)", Gw])
        table.add_row(["  Gini net between groups (Gnb)  ", Gnb])
        table.add_row(["Gini transvariation (Gt)", Gt])
        table.add_row(["Gini total (Gw + Gnb + Gt)", f"{G.item():.4f}"])
        print(table)
        
        if self.method:
            self.correlation_ratio = self.Gini_net_between.item() / self.Gini_total.item()
            table = PrettyTable()
            table.field_names = ["ANOGI", "  Index  "]
            table.add_row(["GMD net between groups (GMDnb)", f"{self.Gini_net_between.item():.4f}"])
            table.add_row(["GMD total", f"{self.Gini_total.item():.4f}"])
            table.add_row(["Correlation ratio (GMDnb / GMD)", f"{self.correlation_ratio:.4f}"])
            print(table)
            
        if self.method is None:
            # Table with Gjh
            group_names = [str(name) for name in self.dataframe[self.group_column].unique()]
            matrix_data = [[float(tensor) for tensor in row] for row in self.matrix_Gij]
            df = pd.DataFrame(matrix_data, index=group_names, columns=group_names)
            styles = [{'selector': 'th.col_heading', 'props': 'text-align: center;'}]
            try:
                styled_df = df.style.set_table_styles(styles)
                styled_df = styled_df.format('{:.4f}')
                print(f"\n{'Matrix of between-group Gini indices G_gh:'}")
                display(styled_df)
            except Exception:
                print(f"\n{'Matrix of between-group Gini indices G_gh:'}")
                display(df)
            
            # Table with Gw
            Gjj_weighted = self.means_tensor * self.tensor_Gjj * self.size_tensor
            Gjj_weighted_data = Gjj_weighted.squeeze().tolist()
            df = pd.DataFrame({group: [value] for group, value in zip(group_names, Gjj_weighted_data)})
            try:
                styled_df = df.style.set_table_styles(styles)
                df.index = ["Weighted Gini within"] * len(df)
                df.reset_index(drop=True, inplace=True)
                print(f"{'Weighted Gini indices p^g*s^g*G_gg:'}")
                styled_df = styled_df.format('{:.4f}')
                display(styled_df)
            except Exception:
                print(f"{'Weighted Gini indices p^g*s^g*G_gg:'}")
                display(df)
                
            # Table with Djh
            group_names = [str(name) for name in self.dataframe[self.group_column].unique()]
            matrix_data = [[float(tensor) for tensor in row] for row in self.matrix_Dij]
            df = pd.DataFrame(matrix_data, index=group_names, columns=group_names)
            styles = [{'selector': 'th.col_heading', 'props': 'text-align: center;'}]
            try:
                styled_df = df.style.set_table_styles(styles)
                styled_df = styled_df.format('{:.4f}')
                print(f"{'Matrix of Distances D_gh:'}")
                display(styled_df)
            except Exception:
                print(f"{'Matrix of Distances D_gh:'}")
                display(df)
            
            # Table with summary (sj, pj, Gjj)
            header = f"{'Summary Table':^45}\n{'=' * 51}\n{'Groups':<15}{'Means':<15}{'Size':<15}{'Gini':<20}\n{'-' * 51}"
            rows = ""
            for i in range(self.num_tensors):
                gini_formatted = f"{self.tensor_Gjj[i].item():.4f}"  
                rows += f"{str(self.dataframe[self.group_column].unique()[i]):<15}{self.means_tensor[i].item():<15.4f}{self.size_tensor[i].item():<15.4f}{gini_formatted}\n"
            print(f"{'=' * 51}\n{header}\n{rows}\n{'=' * 51}")
