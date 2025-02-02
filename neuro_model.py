# import packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
import numpy as np



# read in train and validate set
df = pd.read_csv('Train_and_Validate_EEG.csv')



# drop unnamed column between psd columns and coherence columns
df.drop('Unnamed: 122', axis=1, inplace=True)

psd = df.loc[:, 'AB.A.delta.a.FP1':'AB.F.gamma.s.O2']
psd_aggregate = pd.DataFrame(index=range(len(psd)))
zone_names = ['FP', 'F', 'T', 'C', 'P', 'O']
electrode_names = [['a.FP1', 'b.FP2'], ['c.F7', 'd.F3', 'e.Fz', 'f.F4', 'g.F8'], ['h.T3','l.T4','m.T5','q.T6'], ['i.C3','j.Cz','k.C4',], ['n.P3','o.Pz','p.P4'], ['r.O1','s.O2']]
band_names = ['A.delta', 'B.theta', 'C.alpha', 'D.beta', 'E.highbeta', 'F.gamma']
for band in band_names:
    for j in range(6):
        psd_aggregate['AB.'+band+'.'+zone_names[j]] = 0
        for k in range(len(electrode_names[j])):
            psd_aggregate['AB.'+band+'.'+zone_names[j]] += psd['AB.'+band+'.'+electrode_names[j][k]]
psd_aggregate

coherence = df.loc[:, 'COH.A.delta.a.FP1.b.FP2':]



# aggregate coherences between pairs of regions ###
coherence_hemispheres_aggregate = pd.DataFrame(index=range(len(coherence)))
zone_names = ['FP','F', 'T', 'C', 'P', 'O']
electrode_names = [['a.FP1', 'b.FP2'], ['c.F7', 'f.F4', 'd.F3', 'g.F8', 'e.Fz'], ['h.T3','l.T4','m.T5','q.T6'], ['i.C3','k.C4','j.Cz'], ['n.P3','p.P4','o.Pz'], ['r.O1','s.O2']]
band_names = ['A.delta', 'B.theta', 'C.alpha', 'D.beta', 'E.highbeta', 'F.gamma']

for band in band_names:
    for i in range(6):
        for j in range(i,6):
            coherence_hemispheres_aggregate['COH.hemispheres.'+band+'.'+zone_names[i]+'.'+zone_names[j]] = 0
            for a in range(len(electrode_names[i])):
                for b in range(len(electrode_names[j])):
                    if i!=j:
                        if (a+b)%2 ==1:
                            try:
                                coherence_hemispheres_aggregate['COH.hemispheres.'+band+'.'+zone_names[i]+'.'+zone_names[j]] +=coherence['COH.'+band+'.'+electrode_names[i][a]+'.'+electrode_names[j][b]]
                            except:
                                coherence_hemispheres_aggregate['COH.hemispheres.'+band+'.'+zone_names[i]+'.'+zone_names[j]] +=coherence['COH.'+band+'.'+electrode_names[j][b]+'.'+electrode_names[i][a]]
                    elif(a<b):
                        if (a+b)%2 == 1:
                            try:
                                coherence_hemispheres_aggregate['COH.hemispheres.'+band+'.'+zone_names[i]+'.'+zone_names[j]] +=coherence['COH.'+band+'.'+electrode_names[i][a]+'.'+electrode_names[j][b]]
                            except:
                                coherence_hemispheres_aggregate['COH.hemispheres.'+band+'.'+zone_names[i]+'.'+zone_names[j]] +=coherence['COH.'+band+'.'+electrode_names[j][b]+'.'+electrode_names[i][a]]

coherence_hemispheres_aggregate




# one-hot encode sex
df = pd.get_dummies(df, columns=['sex'], dtype=int, drop_first=True)

new_df = df['main.disorder'].to_frame()
new_df['sex_M'] = df['sex_M']
new_df['age'] = df['age']
new_df['education'] = df['education']
new_df['IQ'] = df['IQ']
new_df = pd.concat([new_df, psd_aggregate, coherence_hemispheres_aggregate], axis=1)




# drop people with n/a for iq and education (only 12 and 13, respectively)
new_df.dropna(subset=["IQ", "education"], inplace=True, ignore_index=True)
y = new_df['main.disorder']
new_df = new_df.drop('main.disorder', axis=1)

print(new_df.shape)


# Separate the dataframe based on conditions
df_mood = new_df[y == "Mood disorder"]
df_healthy = new_df[y == "Healthy control"]
df_schizo = new_df[y == "Schizophrenia"]
df_ocd = new_df[y == "Obsessive compulsive disorder"]
df_addictive = new_df[y == "Addictive disorder"]
df_trauma = new_df[y == "Trauma and stress related disorder"]
df_anxiety = new_df[y == "Anxiety disorder"]

# Perform ANOVA test
f_value, p_value = stats.f_oneway(df_mood, df_healthy, df_schizo, df_ocd, df_addictive, df_trauma, df_anxiety)

# Store p-values in a DataFrame
p_value_df = pd.DataFrame({'Feature': new_df.columns, 'p_value': p_value})

# Ensure filtering works correctly
significant_features = p_value_df.loc[p_value_df['p_value'] < 0.05, 'Feature'].tolist()

# Keep only significant features
new_df = new_df[significant_features]

print(new_df.shape)




#Perform scaling and run regression model
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
target = ['main.disorder']
X = new_df
y = y.values.ravel()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

accuracies = []

for i in range(80, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000, l1_ratio=0.5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Compute aggregation
mean_accuracy = np.mean(accuracies)
std_dev = np.std(accuracies)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")


