# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import LocalOutlierFactor


df = pd.read_pickle(r'../../data/interim/01_data_processed.pkl')

outlier_colums = list(df.columns[:6])


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 100

# similar to hue in seaborn
df[['acc_x','label']].boxplot(by='label')
df[['acc_y','label']].boxplot(by='label')
df[outlier_colums[:3]+['label']].boxplot(by='label',layout = (1,3))
df[outlier_colums[3:]+['label']].boxplot(by='label',layout = (1,3))


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


col = 'acc_x'

dataset = mark_outliers_iqr(df,col)


for col in outlier_colums:
    dataset = mark_outliers_iqr(dataset=dataset,col=col)



df[outlier_colums[:3]+['label']].plot.hist(by='label',layout = (3,3))
df[outlier_colums[3:]+['label']].plot.hist(by='label',layout = (3,3))
  



	
# local outlier detection
	
    
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

dataset,outliers,x_scores = mark_outliers_lof(dataset,columns=outlier_colums)

# chavnets criterion

def find_outliers(values):
    """Return True for each value that is an outlier (Chauvenet's criterion)."""
    mean = values.mean()
    std = values.std()
    n = len(values)

    z = abs(values - mean) / std                  # distance from mean, in std units
    rarity = scipy.special.erfc(z / np.sqrt(2))   # smaller = rarer
    return rarity < 1 / (2 * n)                   # True = outlier


df = df.copy()



df = df.copy()

for col in outlier_colums:
    for label in df["label"].unique():
        rows = df["label"] == label               # True for this exercise's rows
        is_outlier = find_outliers(df.loc[rows, col])

        # is_outlier has the same index as those rows, so this lines up
        outlier_index = is_outlier[is_outlier].index
        df.loc[outlier_index, col] = np.nan

    print(col, "outliers:", df[col].isna().sum())
    


df = df.to_pickle(r'../../data/interim/02_outlier_to_nan.pkl')

# %%
