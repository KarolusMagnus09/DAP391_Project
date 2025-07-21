import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm


def convert_region(df, variableName: str):
    vietnam_regions = {
        "Dong Bang Song Hong": [
            "Ha Noi", "Hai Duong", "Nam Dinh", "Hai Phong"
        ],
        "Trung du va Mien Nui Phia Bac": [
            "Thai Nguyen", "Hoa Binh", "Viet Tri", "Yen Bai",
            "Hong Gai", "Cam Pha", "Uong Bi"
        ],
        "Bac Trung Bo va Duyen Hai Mien Trung": [
            "Thanh Hoa", "Vinh", "Hue", "Tam Ky", "Nha Trang",
            "Qui Nhon", "Tuy Hoa", "Phan Rang", "Cam Ranh"
        ],
        "Tay Nguyen": [
            "Buon Me Thuot", "Da Lat", "Play Cu"
        ],
        "Dong Nam Bo": [
            "Ho Chi Minh City", "Bien Hoa", "Vung Tau", "Phan Thiet"
        ],
        "Dong Bang Song Cuu Long": [
            "Can Tho", "Ben Tre", "Vinh Long", "Tan An",
            "My Tho", "Long Xuyen", "Chau Doc", "Tra Vinh",
            "Bac Lieu", "Ca Mau", "Soc Trang", "Rach Gia"
        ]
    }

    province_to_region = {city: region for region, cities in vietnam_regions.items() for city in cities}
    df[variableName] = df[variableName].map(province_to_region)
    return df

def split_date(df, variableName: str):
  df[variableName] = pd.to_datetime(df[variableName])
  df['day'] = df[variableName].dt.day
  df['month'] = df[variableName].dt.month
  df['year'] = df[variableName].dt.year
  df.drop(columns=[variableName], inplace=True)
  return df

def boxplots(df):
  numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
  plt.figure(figsize=(16, 5))

  for i, column in enumerate(numerical_columns, 1):
      plt.subplot(1, len(numerical_columns), i)
      sns.boxplot(data=df, y=column)
      plt.title(f"Boxplot of {column}")
      plt.ylabel(column)

  plt.tight_layout()
  plt.show()


def findSignificant(df):
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return lower_bound, upper_bound

def Clipping(df):
  lower_bound, upper_bound = findSignificant(df)
  df = df.clip(lower=lower_bound, upper=upper_bound)
  return df


# filtering correlation
def corrFilter(df):
    # Dictionary 
    positive = {'weak': [], 'normal': [], 'good': [], 'strong': [], 'real strong': [], 'perfect': []}
    negative = {'weak': [], 'normal': [], 'good': [], 'strong': [], 'real strong': [], 'perfect': []}
    correlation = df.corr()

    # Iterate through the correlation matrix
    # to categorize pairs of features based on their correlation values
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            corr_value = correlation.iloc[i, j]
            abs_corr = abs(corr_value)
            target_dict = positive if corr_value >= 0 else negative

            if abs_corr < 0.5:
                target_dict['weak'].append(f'{df.columns[i]} - {df.columns[j]}')
            elif 0.5 <= abs_corr < 0.65:
                target_dict['normal'].append(f'{df.columns[i]} - {df.columns[j]}')
            elif 0.65 <= abs_corr < 0.75:
                target_dict['good'].append(f'{df.columns[i]} - {df.columns[j]}')
            elif 0.75 <= abs_corr < 0.8:
                target_dict['strong'].append(f'{df.columns[i]} - {df.columns[j]}')
            elif 0.8 <= abs_corr < 0.9:
                target_dict['real strong'].append(f'{df.columns[i]} - {df.columns[j]}')
            else:
                target_dict['perfect'].append(f'{df.columns[i]} - {df.columns[j]}')

    return positive, negative

def oLs(df, x_var: str, y_var: str):
  model = OLS(df[x_var], sm.add_constant(df[y_var])).fit()
  return model

def ols_metrics(df, x_var: str, y_var: str):
  model = oLs(df, x_var, y_var)

  adj_r_squared = model.rsquared_adj
  f_statistic = model.fvalue
  Prob_F = model.f_pvalue
  return (adj_r_squared, f_statistic, Prob_F)

def anova(df, dl_var: str, dt_var: str):
  model = ols(f'{dl_var} ~ {dt_var}', data= df).fit()
  return model

def anova_metrics(df, dl_var: str, dt_var: str):
  model = anova(df, dl_var, dt_var)
  anova_table = anova_lm(model)

  F_value = anova_table.loc[dt_var, 'F']
  Prob_F = anova_table.loc[dt_var, 'PR(>F)']
  ss_between = anova_table.loc[dt_var, 'sum_sq']
  ss_total = anova_table['sum_sq'].sum()

  eta_squared = ss_between / ss_total
  return (eta_squared, F_value, Prob_F)