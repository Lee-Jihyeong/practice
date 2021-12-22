# 기본 패키지 모음
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 전처리 및 모델링 준비를 위한 패키지
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 실제 모델링을 위한 패키지
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# 모델 평가를 위한 패키지
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# 그래프
def plot_chart(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


df = pd.read_csv('/content/drive/MyDrive/winequality-red.csv')

df.head()

df.info()

df.describe()

df.shape

df.columns.values

plt.figure(figsize=(10, 15))

for i, col in enumerate(list(df.columns.values)):
    plt.subplot(4, 3, i + 1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()

df['quality'].unique()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=df)

# 5.5점을 기준으로 좋은 와인과 나쁜 와인을 구분하겠다고 선언
bins = (2, 5.5, 9)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

# 굿 배드는 인식할 수 없음. 따라서 이를 인식할 수 있도록 Label Encoding을 실시
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])

df['quality'].value_counts()

# 종속변수와 독립변수를 나누어주는 작업
X = df.drop('quality', axis=1)
y = df['quality']

wine_correlation = X.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(wine_correlation, annot=True, cmap="RdBu_r")

plt.figure(figsize=(20, 16))

for i, col in enumerate(list(X.columns.values)):
    plt.subplot(4, 4, i + 1)
    sns.distplot(X[col], color='r', kde=True, label='data')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()

sc = StandardScaler()
X.loc[:, :] = sc.fit_transform(X)

plt.figure(figsize=(20, 16))

for i, col in enumerate(list(X.columns.values)):
    plt.subplot(4, 4, i + 1)
    sns.distplot(X[col], color='r', kde=True, label='data')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()

# 변수별로 Train과 Test 쓸 데이터 셋을 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def show_feature_importance(model, col):
    # 배열형태로 반환
    ft_importance_values = model.feature_importances_

    # 정렬과 시각화를 쉽게 하기 위해 series 전환
    ft_series = pd.Series(ft_importance_values, index=col.columns)
    ft_top10 = ft_series.sort_values(ascending=False)[:10]

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance Top 10')
    sns.barplot(x=ft_top10, y=ft_top10.index)
    plt.show()


# 모델의 선언
svc = SVC(probability=True)  # svc는 prob 뽑으려면 True 해줘야 함

# 학습데이터의 모델 적용
svc.fit(X_train, y_train)

# 모델의 예측값 산출
pred_svc = svc.predict(X_test)

# 다양한 모델 평가
print(classification_report(y_test, pred_svc))

# 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_svc))
print("Precision:", metrics.precision_score(y_test, pred_svc))
print("Recall:", metrics.recall_score(y_test, pred_svc))

# Grid Search
print("Performing grid search ... ")

# Parameter Grid
param_grid = {'C': [0.1, 1.0, 10.0, 100.0],
              'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.00001, 10]}  # 1.0 하고 scale이 defalut

# Make grid search classifier
clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(X_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

# 모델의 선언
svc = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True)  # svc는 prob 뽑으려면 True 해줘야 함

# 학습데이터의 모델 적용
svc.fit(X_train, y_train)

# 모델의 예측값 산출
pred_svc = svc.predict(X_test)

# 다양한 모델 평가
print(classification_report(y_test, pred_svc))

# 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_svc))
print("Precision:", metrics.precision_score(y_test, pred_svc))
print("Recall:", metrics.recall_score(y_test, pred_svc))

probs = svc.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)



print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_svc)))

## 모델의 선언
lr = LogisticRegression()

## 학습데이터의 모델 적용
lr.fit(X_train, y_train)

## 모델의 예측값 산출
pred_lr = lr.predict(X_test)

## 다양한 모델 평가 
print(classification_report(y_test, pred_lr))

## 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_lr))
print("Precision:", metrics.precision_score(y_test, pred_lr))
print("Recall:", metrics.recall_score(y_test, pred_lr))

probs = lr.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plot_chart(fpr, tpr, roc_auc)

print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_lr)))

## 모델의 선언
dtc = tree.DecisionTreeClassifier()

## 학습데이터의 모델 적용
dtc.fit(X_train, y_train)

## 모델의 예측값 산출
pred_dtc = dtc.predict(X_test)

## 다양한 모델 평가 
print(classification_report(y_test, pred_dtc))

## 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_dtc))
print("Precision:", metrics.precision_score(y_test, pred_dtc))
print("Recall:", metrics.recall_score(y_test, pred_dtc))

probs = dtc.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plot_chart(fpr, tpr, roc_auc)

print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_dtc)))

## 모델의 선언
rfc = RandomForestClassifier(n_estimators=200)

## 학습데이터의 모델 적용
rfc.fit(X_train, y_train)

## 모델의 예측값 산출
pred_rfc = rfc.predict(X_test)

## 다양한 모델 평가 
print(classification_report(y_test, pred_rfc))

## 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_rfc))
print("Precision:", metrics.precision_score(y_test, pred_rfc))
print("Recall:", metrics.recall_score(y_test, pred_rfc))

show_feature_importance(rfc, X)

probs = rfc.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plot_chart(fpr, tpr, roc_auc)

print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_rfc)))

## 모델의 선언
knn = KNeighborsClassifier(5)

## 학습데이터의 모델 적용
knn.fit(X_train, y_train)

## 모델의 예측값 산출
pred_knn = knn.predict(X_test)

## 다양한 모델 평가 
print(classification_report(y_test, pred_knn))

## 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_knn))
print("Precision:", metrics.precision_score(y_test, pred_knn))
print("Recall:", metrics.recall_score(y_test, pred_knn))

probs = knn.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plot_chart(fpr, tpr, roc_auc)

print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_knn)))

## 모델의 선언
nb = GaussianNB()

## 학습데이터의 모델 적용
nb.fit(X_train, y_train)

## 모델의 예측값 산출
pred_nb = nb.predict(X_test)

## 다양한 모델 평가 
print(classification_report(y_test, pred_nb))

## 가장 기초적인 성적 평가 지표들
print("Accuracy:", metrics.accuracy_score(y_test, pred_nb))
print("Precision:", metrics.precision_score(y_test, pred_nb))
print("Recall:", metrics.recall_score(y_test, pred_nb))

probs = nb.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plot_chart(fpr, tpr, roc_auc)

print('AUC 면적 : {}'.format(roc_auc_score(y_test, pred_nb)))
