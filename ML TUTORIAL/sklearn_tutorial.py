# import sklearn
import pandas as pd
import numpy as np
# from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
# from sklearn.feature_selection import SelectKBest,VarianceThreshold,mutual_info_classif,f_classif
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC,SVR,OneClassSVM
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
# from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor,BaggingClassifier,IsolationForest
# from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# data = {'worked_years':[1,2,3,4,5],
#         'salary':[1000,2000,3000,400,5000]
#         }

# df = pd.DataFrame(data)
# print(df)


# df = pd.read_csv('Titanic-Dataset.csv')

# print(df)

# data = {
#     'Gender': [1,0,0,0],
#     'Age': [23, 35, 29, 40],
#     'Salary': [50000, 60000, 55000, 80000],
#     'Color': [2,0,1,0],
#     'Expirence':[1,3,2,4]
# }
# df = pd.DataFrame(data)
# print('ORIGINAL DF \n',df)

# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]


# model = mutual_info_classif(X,y,discrete_features=True)

# for feature,score in zip(df.columns,model):
#     print(feature,score)

# output = model.fit_transform(X,y)
# print(output)

# sel_col = model.get_support(indices=True)

# new_df = pd.DataFrame(output,columns = [df.columns[x] for x in sel_col ])
# print(new_df)



# data = {
#     'Gender': [1,0,0,0],
#     'Age': [23, 35, 29, 40],
#     'Salary': [50000, 60000, 55000, 80000],
#     'Color': [2,0,1,0],
#     'Expirence':[1,3,2,4]
# }
# df = pd.DataFrame(data)
# print('ORIGINAL DF \n',df,'\n')

# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_train)
# print(y_train)

# print('\n')
# print(X_test)
# print(y_test)



# data = {
#     'worked_year': [1,2,3,4,5],
#     'Salary': [30000, 5000, 40000, 45000,50000]}


# df = pd.DataFrame(data)

# predict_df = pd.DataFrame({'worked_year': [1,2,3,4,5]})
# # print(df)

# X = df[['worked_year']]
# y = df['Salary']


# model = DecisionTreeRegressor()

# model.fit(X,y)

# y_pred = model.predict(predict_df[['worked_year']])
# print(y_pred)

# print('MSE :',mean_squared_error(y,y_pred))
# print('MAE :',mean_absolute_error(y,y_pred))
# print('R2 score :',r2_score(y,y_pred))


# plt.scatter(df['worked_year'],df['Salary'],color='red',label='Actual')
# plt.plot(predict_df['worked_year'],y_pred,color='blue',label='Predicted')
# plt.legend()
# plt.show()









# data = {
#     'Age': [18, 22, 25, 28, 30, 35, 40, 45],
#     'Salary': [11000, 2000, 2500, 38000, 4000, 4500, 5000, 600],
#     'Purchased': [0, 0, 0, 1, 1, 1, 1, 1]  # 1 means bought, 0 means not bought
# }

# df = pd.DataFrame(data)

# data2 = {
#     'Age': [2, 282, 265, 128, 309, 0, 4, 45],
#     'Salary': [11, 2000, 250, 38000, 200, 400, 50, 600]}
# pred_df = pd.DataFrame(data2)

# X = df[['Age','Salary']]
# y = df['Purchased']

# # df = load_iris()
# # X = df.data
# # y = df.target

# # print(df.feature_names)
# # print('X: \n',X)
# # print('y: \n',y)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model = DecisionTreeClassifier()
# model.fit(X_train,y_train)

# y_pred = model.predict(pred_df)
# print(y_pred)

# preds = model.predict_proba(pred_df)
# print(preds)
# # print('X TEST : \n',X_test)

# # print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# # print('Confusion Metrix :',confusion_matrix(y_test,y_pred))
# # print('Classification Report :',classification_report(y_test,y_pred))


# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'EducationLevel': ['HighSchool','Diploma','Diploma','BSc','BSc','MSc','MSc','PhD','PhD','BSc','MSc','HighSchool','Diploma','PhD','MSc'],
#     'City': ['Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano'],
#     'Department': ['IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance'],
#     'Gender': ['Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male'],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }

# df = pd.DataFrame(classification_data)
# X = df[['Age','Salary','YearsExperience','EducationLevel','City','Department','Gender']]
# y = LabelEncoder().fit_transform(df['Target'])

# ordinal_col = ['EducationLevel','Gender']
# onehot_col = ['City','Department']


# preprocess = ColumnTransformer([('ord_enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),ordinal_col),
#                                 ('onehot_enc',OneHotEncoder(handle_unknown='ignore'),onehot_col),
#                                 ])

# model = Pipeline([('process',preprocess),
#                   ('standard',StandardScaler()),
#                   ('tree',DecisionTreeClassifier())
#                   ])

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# model.fit(X_train,y_train)



# y_pred = model.predict(X_test)
# print(y_pred)


# print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# print('Confusion Metrix :',confusion_matrix(y_test,y_pred))
# print('Classification Report :',classification_report(y_test,y_pred))


# # joblib.dump(model,'trained_model.joblib')
# # model = joblib.load('trained_model.joblib')

# # file = open('trained_model_pickle.pkl','wb')
# # pickle.dump(model,file)

# file = open('trained_model_pickle.pkl','rb')
# model = pickle.load(file)


# y_pred = model.predict([[200,28,11]])
# print(y_pred)


# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }

# df = pd.DataFrame(classification_data)


# X = df[['Age','Salary','YearsExperience']]
# y = LabelEncoder().fit_transform(df['Target'])

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train,y_train)



# y_pred = model.predict([[100,50,50]])
# print(y_pred)


# print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# print('Confusion Metrix :',confusion_matrix(y_test,y_pred))
# print('Classification Report :',classification_report(y_test,y_pred))



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

# model = AdaBoostRegressor()
# model.fit(X_train,y_train)


# y_pred = model.predict(X_test)
# # print(X_test,'\n',y_test)
# print(y_pred)

# print('MSE :',mean_squared_error(y_test,y_pred))
# print('MAE :',mean_absolute_error(y_test,y_pred))
# print('R2 score :',r2_score(y_test,y_pred))


# # # plt.scatter(X_test.iloc[:,0],y_test,color='red',label='Actual')
# # # plt.scatter(X_test.iloc[:,0],y_pred,color='blue',label='Predicted')
# # # plt.legend()
# # # plt.show()




# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }

# df = pd.DataFrame(classification_data)
# print(df)


# df = load_iris()
# X = df.data
# y = df.target
# # print(X.features_names)
# print(X)
# print('\n')
# print(y)
# # X = df[['Age','Salary','YearsExperience']]
# y = LabelEncoder().fit_transform(df['Target'])

# tss = TimeSeriesSplit(n_splits=3)


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# base_model = DecisionTreeClassifier()

# model = AdaBoostClassifier()
# model.fit(X_train,y_train)


# y_pred = model.predict(X_test)
# print(X_test,'\n',y_test)
# print('PREDICTED RESULT :',y_pred)


# print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# print('Confusion Metrix :\n',confusion_matrix(y_test,y_pred))
# print('Classification Report :\n',classification_report(y_test,y_pred))



# for train_row,test_row in tss.split(X):
#     # print('train_row:::\n',train_row)
#     # print('test_row:::\n',test_row)

#     X_train,X_test = X.iloc[train_row],X.iloc[test_row]
#     y_train,y_test = y[train_row],y[test_row]
#     # print(X_train,X_test)
#     # print(y_train,y_test)

#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
#     print(y_pred)


#     print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
#     # print('Confusion Metrix :',confusion_matrix(y_test,y_pred))
    # print('Classification Report :',classification_report(y_test,y_pred))





# unsupervised_data = {
#     'Age': [22,25,27,30,32,35,37,40,42,45,47,50,52,55,58],
#     'Salary': [30000,35000,37000,42000,45000,48000,52000,58000,60000,63000,65000,68000,70000,72000,75000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'SalaryIncrease': [31000,36000,38000,43000,46000,49000,53000,59000,60500,64000,66000,69000,71000,73000,76000]
# }

# df = pd.DataFrame(unsupervised_data)
# # print(df)


# model = AgglomerativeClustering(n_clusters=4) #0,1,2,3
# model.fit(df)
# df['Cluster'] = model.fit_predict(df)
# print(df)

# score = silhouette_score(df,model.labels_)

# print('SCORE :',score)





# unsupervised_data = {
#     'Age': [22,25,27,30,32,35,37,40,42,45,47,50,52,55,58],
#     'Salary': [30000,35000,37000,42000,45000,48000,52000,58000,60000,63000,65000,68000,70000,72000,75000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'SalaryIncrease': [31000,36000,38000,43000,46000,49000,53000,59000,60500,64000,66000,69000,71000,73000,76000]
# }

# df = pd.DataFrame(unsupervised_data)
# # print(df)

# model = OneClassSVM()
# model.fit(df)

# pred = model.predict(df)
# score = (model.decision_function(df)).round(6)
# df['pred'] = pred
# df['score'] = score
# print(df)





# unsupervised_data = {
#     'Age': [22,25,27,30,32,35,37,40,42,45,47,50,52,55,58],
#     'Salary': [30000,35000,37000,42000,45000,48000,52000,58000,60000,63000,65000,68000,70000,72000,75000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'SalaryIncrease': [31000,36000,38000,43000,46000,49000,53000,59000,60500,64000,66000,69000,71000,73000,76000]
# }

# df = pd.DataFrame(unsupervised_data)
# # print('original \n',df)

# scaled_model = StandardScaler()
# df_scaled = scaled_model.fit_transform(df)
# # print('scaled \n',df_scaled)


# model = PCA(n_components=4)
# pred = model.fit_transform(df_scaled)
# print(pred)

# print((model.explained_variance_ratio_).round(2))

# selected_feature = pd.DataFrame(model.components_,columns=['PC1','PC2','PC3','PC4'])
# print(selected_feature)


# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }
# classification_data['Target'] = LabelEncoder().fit_transform(classification_data['Target'])
# df = pd.DataFrame(classification_data)
# print(df)

# X = df[['Age','Salary','YearsExperience']]
# y = df['Target']

# splitter = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# score = cross_val_score(DecisionTreeClassifier(),X,y,cv=splitter)

# print('SCORE :',score)
# print('MEAN SCORE :',score.mean())



# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }
# classification_data['Target'] = LabelEncoder().fit_transform(classification_data['Target'])
# df = pd.DataFrame(classification_data)
# # print(df)

# X = df[['Age','Salary','YearsExperience']]
# y = df['Target']

# param = {
#     'max_depth':[1,4,67,23,5,7,8],
#     'min_samples_split':[2,4,23],
#     'min_samples_leaf':[4,1]
# }
# #max_depth=1, min_samples_leaf=4, min_samples_split=23
# model = DecisionTreeClassifier()

# # splitter = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# # score = cross_val_score(model,X,y,cv=splitter)

# # print('SCORE :',score)
# # print('MEAN SCORE :',score.mean())


# grid = RandomizedSearchCV(model,param,cv=5)
# grid.fit(X,y)

# print('BEST PARAMETER :',grid.best_params_)
# print('BEST SCORE :',grid.best_score_)

# [0,1,1,0,0,0,0,0,0,0]


# classification_data = {
#     'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
#     'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
#     'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
# }
# classification_data['Target'] = LabelEncoder().fit_transform(classification_data['Target'])
# df = pd.DataFrame(classification_data)
# print(df)

# X = df[['Age','Salary','YearsExperience']]
# y = df['Target']

# # print('INITIAL Y COUNT :',Counter(y))

# # model = RandomUnderSampler()
# # X_res,y_res = model.fit_resample(X,y)

# # print('SECOND Y COUNT :',Counter(y_res))



# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# model = DecisionTreeClassifier(class_weight={1:3,0:7} )
# model.fit(X_train,y_train)


# y_pred = model.predict(X_test)
# print(X_test,'\n',y_test)
# print('PREDICTED RESULT :',y_pred)


# print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# print('Confusion Metrix :\n',confusion_matrix(y_test,y_pred))
# print('Classification Report :\n',classification_report(y_test,y_pred))



# text = ["Buy cheap shoes now",
#         "Cheap shoes available today"]


# model = TfidfVectorizer()
# pred = model.fit_transform(text)
# print(text)
# print('VOCABULARY :',model.get_feature_names_out())
# print(pred.toarray())


df = pd.read_csv('Email_Spam_Dataset.csv')
print(df.head())

X = df['text']
y = df['label_num']

print(Counter(y))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(class_weight ='balanced')

vect = TfidfVectorizer()
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

my_data =  vect.transform(['buy today for free','ValueError: Iterable over raw text documents expected, string object received.'])
model.fit(X_train_vect,y_train)

y_pred = model.predict(my_data)

# print(X_test,'\n',y_test)
print('PREDICTED RESULT :',y_pred)


# print('Accuracy Score :',accuracy_score(y_test,y_pred)*100)
# print('Confusion Metrix :\n',confusion_matrix(y_test,y_pred))
# print('Classification Report :\n',classification_report(y_test,y_pred))