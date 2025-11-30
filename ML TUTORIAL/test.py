# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Bagging
# bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20)
# bag.fit(X_train, y_train)
# bag_pred = bag.predict(X_test)

# # Boosting
# boost = AdaBoostClassifier(n_estimators=50, learning_rate=0.7)
# boost.fit(X_train, y_train)
# boost_pred = boost.predict(X_test)

# print("Bagging Accuracy =", accuracy_score(y_test, bag_pred))
# print("Boosting Accuracy =", accuracy_score(y_test, boost_pred))




# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans,AgglomerativeClustering
# from sklearn.metrics import silhouette_score

# np.random.seed(42)

# # ---- Cluster 1: Young low-income tech users ----
# cluster1 = pd.DataFrame({
#     "Age": np.random.normal(23, 3, 50),                 # young
#     "Income": np.random.normal(35000, 4000, 50),        # low income
#     "SpendingScore": np.random.normal(60, 8, 50),       # moderate spending
#     "OnlineHours": np.random.normal(5, 1, 50),          # high online hours
#     "Transactions": np.random.normal(20, 3, 50),        # medium
#     "VisitsPerMonth": np.random.normal(10, 2, 50)       # high
# })

# # ---- Cluster 2: Middle-age high-income professionals ----
# cluster2 = pd.DataFrame({
#     "Age": np.random.normal(40, 4, 50),                 # middle age
#     "Income": np.random.normal(95000, 8000, 50),        # high income
#     "SpendingScore": np.random.normal(30, 5, 50),       # low spending score
#     "OnlineHours": np.random.normal(2, 0.5, 50),        # low online hours
#     "Transactions": np.random.normal(10, 2, 50),        # low
#     "VisitsPerMonth": np.random.normal(4, 1, 50)        # low
# })

# # ---- Cluster 3: Older moderate-income shoppers ----
# cluster3 = pd.DataFrame({
#     "Age": np.random.normal(60, 5, 50),                 # older
#     "Income": np.random.normal(50000, 6000, 50),        # mid income
#     "SpendingScore": np.random.normal(80, 7, 50),       # high spending score
#     "OnlineHours": np.random.normal(3, 1, 50),          # moderate hours
#     "Transactions": np.random.normal(25, 4, 50),        # high
#     "VisitsPerMonth": np.random.normal(12, 3, 50)       # high visits
# })

# print(cluster1.head())
# print(cluster2.head())
# print(cluster3.head())
# # Merge all clusters
# df = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

# # Make numbers neat
# df = df.round(2)
# print('main df')
# # print(df.to_string())
# # print(df.shape)


# model = AgglomerativeClustering(n_clusters=3)
# model.fit(df)
# df["Cluster"] = model.fit_predict(df)
# print(df.head())

# score = silhouette_score(df,model.labels_)
# print(score)

# from sklearn.cluster import KMeans,AgglomerativeClustering
# import pandas as pd

# df = pd.DataFrame({
#     "Age":  [25, 24, 26, 40, 42, 39, 60, 59, 11],
#     "Income": [30000, 32000, 29000, 90000, 87000, 88000, 50000, 52000, 4000]
# })

# model = AgglomerativeClustering(n_clusters=2)

# labels = model.fit_predict(df)
# print(labels)


from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.svm import OneClassSVM

# # Example: customers and their monthly spending
# df = pd.DataFrame({
#     "Spending": [100, 120, 130, 110, 125, 140, 100000,100200]  # last value is abnormal
# })

# df = pd.DataFrame({
#     "Age":  [25, 24, 26, 40, 42, 39, 60, 59, 11],
#     "Income": [30000, 32000, 29000, 90000, 87000, 88000, 50000, 52000, 4000]
# })


# df = pd.DataFrame({
#     "Age":     [21,22,23,24,25,26,27,28,80],
#     "Income":  [30000,31000,32000,30500,33000,34000,35000,36000,900000],
#     "Spending":[100,120,130,110,125,140,150,160,1000],
#     "Experience":[1,2,3,4,5,6,7,8,50],
#     "Visits":  [5,5,6,6,5,5,6,6,100],
#     "Score":   [70,72,68,75,71,73,74,76,10]
# })
# model = IsolationForest()
# labels = model.fit_predict(df)

# print(labels)  # -1 = anomaly, 1 = normal



# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import pandas as pd





# # Load dataset
# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)

# df = pd.DataFrame({
#     "Age":     [21,22,23,24,25,26,27,28,80],
#     "Income":  [30000,31000,32000,30500,33000,34000,35000,36000,900000],
#     "Spending":[100,120,130,110,125,140,150,160,1000],
#     "Experience":[1,2,3,4,5,6,7,8,50],
#     "Visits":  [5,5,6,6,5,5,6,6,100],
#     "Score":   [70,72,68,75,71,73,74,76,10]
# })
# # Step 1: Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df)

# # Step 2: Apply PCA (2 components)
# pca = PCA(n_components=2 )
# X_pca = pca.fit_transform(X_scaled)
# print(X_pca)

# # Create new dataframe
# # pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# # print("Original Shape:", df.shape)
# # print("PCA Shape:", pca_df.shape)
# # print(pca_df.head())
# print(pca.explained_variance_ratio_)


# # print("Total Variance Explained:", sum(pca.explained_variance_ratio_))
# loadings = pd.DataFrame(
#     pca.components_,
#     columns=df.columns
# )

# print(loadings)


# from sklearn.inspection import permutation_importance
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression,LogisticRegression
# from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


# regression_data = {
#     'Age': [22,25,27,30,32,35,37,40,42,45,47,50,52,55,58],
#     'Salary': [30000,35000,37000,42000,45000,48000,52000,58000,60000,63000,65000,68000,70000,72000,75000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': [31000,36000,38000,43000,46000,49000,53000,59000,60500,64000,66000,69000,71000,73000,76000]
# }


# df = pd.DataFrame(regression_data)
# # print(df)

# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model = DecisionTreeClassifier()
# model.fit(X_train,y_train)

# # print(model.coef_)

# print(model.feature_importances_)


# r = permutation_importance(model, X_test, y_test, n_repeats=20)

# print("Feature Importance:")
# for feature, score in zip(X.columns, r.importances_mean):
#     print(feature, ":", score)


# from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier


# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }

# classification_data['Target'] = LabelEncoder().fit_transform(classification_data['Target'])
# df = pd.DataFrame(classification_data)
# print(df)

# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]

# kfold = KFold(n_splits=5,shuffle=True,random_state=42)
# print(kfold)
# # score = cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
# # print(score)
# # print(score.mean())


# data = {
#     'email': [
#         "Win a brand new car now!",
#         "Limited time offer, claim your prize",
#         "Hey buddy, how are you doing today?",
#         "Meeting at 10am tomorrow",
#         "You’ve been selected to receive $1000",
#         "Let’s grab lunch today"
#     ],
#     'label': [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
# }

# df = pd.DataFrame(data)


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

classification_data = {
    'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
    'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
    'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
}

classification_data['Target'] = LabelEncoder().fit_transform(classification_data['Target'])
df = pd.DataFrame(classification_data)
# print(df)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(Counter(y))

sm = RandomUnderSampler()
X_res, y_res = sm.fit_resample(X, y)

print(Counter(y_res))
