# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

plt.style.use('fivethirtyeight')

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(r"../../data/interim/02_outlier_to_nan.pkl")

predictor_cols = list(df.columns[:6])

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_cols:
    df[col] = df[col].interpolate()

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for s in df['set'].sort_values().unique():
    start = df[df['set'] == s].index[0]
    end = df[df['set'] == s].index[-1]
    duration = (end - start)
    df.loc[(df['set'] == s),'duration'] = duration.seconds
    

df.groupby(['category'])['duration'].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3


for col in predictor_cols:
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col,fs,cutoff)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()


pc_values = PCA.determine_pc_explained_variance(df_pca,predictor_cols)
pd.Series(predictor_cols,pc_values)

# choosing optimum no of componenes using elbow methods
# explained_variance v/s component number

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_cols) + 1),pc_values)
plt.xlabel('Componenet Number')
plt.ylabel('Explained Variance')
plt.show()

df_applied_pca = PCA.apply_pca(df_pca,predictor_cols,3)


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_applied_pca.copy()

acc_r = (df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z'] ** 2)
gyro_r = (df_squared['gyro_x'] ** 2 + df_squared['gyro_y'] ** 2 + df_squared['gyro_z'] ** 2)

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyro_r'] = np.sqrt(gyro_r)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_cols = predictor_cols + ['acc_r' , 'gyro_r']

ws = 5  # your window size

roll = df_temporal.groupby("set")[predictor_cols].rolling(ws)

mean = roll.mean().reset_index(level=0, drop=True)
std = roll.std().reset_index(level=0, drop=True)

mean.columns = [f"{c}_temp_mean_ws_{ws}" for c in predictor_cols]
std.columns = [f"{c}_temp_std_ws_{ws}" for c in predictor_cols]

df_temporal = pd.concat([df_temporal, mean, std], axis=1)	  
    

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800 / 200)

df_freq_list = []

for s in list(df_freq["set"].unique()):
    print(f'Applying for set {s}')
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset,predictor_cols,ws,fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list)

freq_cols = [c for c in df_freq.columns if "_freq" in c or "_pse" in c]
roll_cols = [c for c in df_freq.columns if "_temp_" in c]

expected_base = (
    list(df_freq.columns[:6])
    + ["participant", "label", "category", "set", "duration"]
    + ["pca_1", "pca_2", "pca_3", "acc_r", "gyro_r"]
)

extra = [c for c in df_freq.columns
         if c not in freq_cols + roll_cols + expected_base]

print(len(df_freq.columns), len(freq_cols), len(roll_cols))
print(extra)

df_freq = df_freq.drop(columns=extra)
print(df_freq.shape)

df_freq = df_freq.set_index("epoch (ms)")

df_freq = df_freq.loc[:, ~df_freq.columns.duplicated()]

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()

df_freq =df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_cols = ['acc_x','acc_y','acc_z']
k_values = range(2,10)
intertia = []

for k in k_values:
    subset = df_cluster[cluster_cols]
    kmeans = KMeans(n_clusters=k,n_init=20)
    cluster_labels = kmeans.fit_predict(subset)
    intertia.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=5,n_init=20)
subset = df_cluster[cluster_cols]
df_cluster['cluster'] = kmeans.fit_predict(subset)
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle('../../data/interim/03_data_features.pkl')

# %%
