# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.model_selection import KFold, GridSearchCV
import time
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict


# 드라이브에서 읽기
iphone = pd.read_csv("/content/drive/MyDrive/iBuy.csv")

# 변수삭제, 변수조정
iphone = iphone.drop(columns=['SERIAL'])
iphone['HOUSEHOLD INCOME/Mo.']=np.log(iphone['HOUSEHOLD INCOME/Mo.'])
iphone['AGE']=np.log(iphone['AGE'])

# 변수확인
iphone.head(5)

# 원본을 남겨두고 복사본 생성
iphone_ = iphone.copy()
# 복사본 내용 셔플
iphone_ = sklearn.utils.shuffle(iphone_)
# 복사본을 훈련세트, 검증세트로 분할 
# 총 426개, 훈련세트: 284, 검증세트: 142
iphone__ = iphone_.iloc[:284]
iphone_test = iphone_.iloc[284:]
# X, Y는 훈련세트, x, y는 검증세트
X = iphone__.iloc[:,:-1]
Y = iphone__.iloc[:,-1]
x = iphone_test.iloc[:,:-1]
y = iphone_test.iloc[:,-1]

# K-fold 5개로 나눔
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
# gridsearch를 위한 파라미터 설정
param_grid = [{'base_model__kernel': ['linear'],
              'base_model__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]#,
              #'base_model__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              #'base_model__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              },
              {'base_model__kernel': ['poly'],
              'base_model__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'base_model__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'base_model__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              },
              {'base_model__kernel': ['rbf'],
              'base_model__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'base_model__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]#,
              #'base_model__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              },
              {'base_model__kernel': ['sigmoid'],
              'base_model__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'base_model__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]#,
              #'base_model__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              }]

"""##Scaler(Standard, MinMax, MaxAbs, Robust) + SVC + GridSearch + 평가, 혼동행렬, ROC_AUC(학습, 테스트, 검증)"""

# 데이터셋을 훈련, 테스트 셋으로 나눔
X_, x_, Y_, y_ = train_test_split(X, Y, test_size=0.33, random_state=1)
# Scaler와 SVC를 동시에 사용하기 위한 파이프라인 설정
clf = Pipeline([('scaler', StandardScaler()),('base_model',SVC(random_state=1))])
"""
clf = Pipeline([('scaler', MinMaxScaler()),('base_model',SVC(random_state=1))])
clf = Pipeline([('scaler', MaxAbsScaler()),('base_model',SVC(random_state=1))])
clf = Pipeline([('scaler', RobustScaler()),('base_model',SVC(random_state=1))])
"""
# 모델 학습
clf.fit(X_, Y_)
# 학습 결과 출력, 정확도와 ROC_AUC, 훈련-테스트-검증셋 전부 출력

print('학습평가: ', clf.score(X_, Y_))
print('교차검증 학습평가: ', round(np.mean(cross_val_score(clf, X_, Y_, cv=kfold)),2))
print('학습 ROC AUC: ', roc_auc_score(Y_, clf.predict(X_)))
print('교차검증 학습 ROC AUC: ', roc_auc_score(Y_, cross_val_predict(clf, X_, Y_, cv=kfold)))
print('Confusion Metrics(Train):')
print(classification_report(Y_, clf.predict(X_)))
print('교차검증 Confusion Metrics(Train):')
print(classification_report(Y_, cross_val_predict(clf, X_, Y_, cv=kfold)))

print('테스트평가: ', clf.score(x_, y_))
print('교차검증 테스트평가: ', round(np.mean(cross_val_score(clf, x_, y_, cv=kfold)),2))
print('테스트 ROC AUC: ', roc_auc_score(y_, clf.predict(x_)))
print('교차검증 테스트 ROC AUC: ', roc_auc_score(y_, cross_val_predict(clf, x_, y_, cv=kfold)))
print('Confusion Metrics(Test):')
print(classification_report(y_, clf.predict(x_)))
print('교차검증 Confusion Metrics(Test):')
print(classification_report(y_, cross_val_predict(clf, x_, y_, cv=kfold)))

print('검증평가: ', clf.score(x, y))
print('교차검증 검증평가: ', round(np.mean(cross_val_score(clf, x, y, cv=kfold)),2))
print('검증 ROC AUC: ', roc_auc_score(y, clf.predict(x)))
print('교차검증 검증 ROC AUC: ', roc_auc_score(y, cross_val_predict(clf, x, y, cv=kfold)))
print('Confusion Metrics(Validation):')
print(classification_report(y, clf.predict(x)))
print('교차검증 Confusion Metrics(Validation):')
print(classification_report(y, cross_val_predict(clf, x, y, cv=kfold)))

# 데이터셋을 훈련, 테스트 셋으로 나눔
X_, x_, Y_, y_ = train_test_split(X, Y, test_size=0.33, random_state=1)
# Scaler와 SVC를 동시에 사용하기 위한 파이프라인 설정
clf = Pipeline([('scaler', StandardScaler()),('base_model',SVC(random_state=1))])
"""
clf = Pipeline([('scaler', MinMaxScaler()),('base_model',SVC(random_state=1))])
clf = Pipeline([('scaler', MaxAbsScaler()),('base_model',SVC(random_state=1))])
clf = Pipeline([('scaler', RobustScaler()),('base_model',SVC(random_state=1))])
"""

# 시간출력 + Gridsearch - 위에서 설정한 k-fold와 파라미터를 사용
start_time = time.time()
grid_model = GridSearchCV(estimator=clf, param_grid=param_grid,
                          cv=kfold, n_jobs=-1, verbose=0).fit(X_, Y_)
print('시간: ',time.time()-start_time)

# 최적 학습 결과 출력, 정확도와 ROC_AUC, 훈련-테스트-검증셋 전부 출력
print('교차검증 점수: ', grid_model.best_score_)
print('최적의 하이퍼 파라메터 조합: ', grid_model.best_params_)
print('최적의 모델: ', grid_model.best_estimator_)
print('학습평가: ', grid_model.score(X_, Y_))
print('학습 ROC AUC: ', roc_auc_score(Y_, grid_model.predict(X_)))
print('Confusion Metrics(Train):')
print(classification_report(Y_, grid_model.predict(X_)))
print('테스트평가: ', grid_model.score(x_, y_))
print('테스트 ROC AUC: ', roc_auc_score(y_, grid_model.predict(x_)))
print('Confusion Metrics(Test):')
print(classification_report(y_, grid_model.predict(x_)))
print('검증 평가: ', grid_model.score(x, y))
print('검증 ROC AUC: ', roc_auc_score(y, grid_model.predict(x)))
print('Confusion Metrics(Validation):')
print(classification_report(y, grid_model.predict(x)))
# 필요하다면 Outer k-fold를 사용해 이 과정을 K번 진행할 수 있다.

# Commented out IPython magic to ensure Python compatibility.
means = grid_model.cv_results_['mean_test_score']
stds = grid_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
print()

print(grid_model.best_params_)

"""##Decision Tree and Plot"""

# 데이터셋을 훈련, 테스트 셋으로 나눔
X_, x_, Y_, y_ = train_test_split(X, Y, test_size=0.33, random_state=1)
# DecisionTree 분류기를 선언 후 학습
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_, Y_)
# 학습 결과 출력, 정확도와 ROC_AUC, 훈련-테스트-검증셋 전부 출력
print('학습평가: ', clf.score(X_, Y_))
print('학습 ROC AUC: ', roc_auc_score(Y_, clf.predict(X_)))
print('Confusion Metrics(Train):')
print(classification_report(Y_, clf.predict(X_)))
print('테스트평가: ', clf.score(x_, y_))
print('테스트 ROC AUC: ', roc_auc_score(y_, clf.predict(x_)))
print('Confusion Metrics(Test):')
print(classification_report(y_, clf.predict(x_)))
print('검증평가: ', clf.score(x, y))
print('검증 ROC AUC: ', roc_auc_score(y, clf.predict(x)))
print('Confusion Metrics(Validation):')
print(classification_report(y, clf.predict(x)))

# 앞서 학습시킨 DecisionTree 분류기를 plot으로 표현
# fn은 column name, cn은 target값
fn = X.columns
cn = ['0','1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,feature_names = fn, 
               class_names=cn,
               filled = True)

"""
#Classifiers, ensemble, voting
"""
# 앙상블 모델
# Voting 알고리즘
# to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels - Sklearn 0.24.2, 1.11.6 Voting Classifier
# 우선 서로다른 7개의 모델을 선언
clf1 = tree.DecisionTreeClassifier()
clf2 = RandomForestClassifier(random_state=0)
clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
clf4 = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
clf5 = Pipeline([('scaler', StandardScaler()),('base_model',SVC(C=100,degree=1,gamma=0.01,random_state=1))])
clf6 = BernoulliNB()
clf7 = LogisticRegression(random_state=0)
# Voting 분류기가 estimators에 있는 분류기들을 호출
eclf = VotingClassifier(estimators=[('clf1',clf1),('clf2',clf2),('clf3',clf3),('clf4',clf4),('clf5',clf5),('clf6',clf6),('clf7',clf7)],voting='hard')
eclf.fit(X_, Y_)

print('학습평가: ', eclf.score(X_, Y_))
print('학습 ROC AUC: ', roc_auc_score(Y_, eclf.predict(X_)))
print('Confusion Metrics(Train):')
print(classification_report(Y_, eclf.predict(X_)))
print('테스트평가: ', eclf.score(x_, y_))
print('테스트 ROC AUC: ', roc_auc_score(y_, eclf.predict(x_)))
print('Confusion Metrics(Test):')
print(classification_report(y_, eclf.predict(x_)))
print('검증평가: ', eclf.score(x, y))
print('검증 ROC AUC: ', roc_auc_score(y, eclf.predict(x)))
print('Confusion Metrics(Validation):')
print(classification_report(y, eclf.predict(x)))