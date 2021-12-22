#!/usr/bin/env python
# coding: utf-8
# created by 데이터 응용개발팀 이지형
# # 1. 라이브러리 import
# Gensim, Konlpy 등의 패키지가 없는 경우 pip 등 을 사용하여 설치
# -*- coding: UTF-8 -*-
import os       # 디렉토리, 파일 탐색 및 이용
import json     # json 파일 사용
import copy     # deepcopy 사용
import re       # 정규식 사용
from tqdm import tqdm   # for 문 진행상황 확인
import multiprocessing  # cpu 코어계산
from collections import namedtuple  # namedtuple 자료형 사용
from gensim.models import doc2vec   # doc2vec 임베딩
import random                       # shuffle 기능
from konlpy.tag import Okt          # 한글 형태소 분석
# ########### scikit learn 의 머신러닝 알고리즘 및 f1 score 계산
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
# ###########
from joblib import dump, load           # fit 한 머신러닝 모델 저장 및 불러오기
from matplotlib import pyplot as plt    # 그래프 그리기
import numpy as np                      # 수학적 계산 및 자료형


# # 2. 클래스 설계

# ## 2.1 머신러닝 모델 클래스

class ML_Model:
    """
    Machine Learning 모델 생성 클래스.
    """
    def __init__(self): # 생성자, 모델 생성 및 저장과 불러오기를 위한 기본 변수 설정
        """
        ML_Model 생성자.
        LogisticRegression, SVC SVM, LinearSVC SVM, BernoulliNB, GaussianNB, MLPClassifier
        총 6개의 모델 생성 및 초기화
        저장과 불러오기를 위한 변수들인 '모델명'_modelfile를 초기화 
        """
        # 기본 6개의 모델 생성
        self.lgr = LogisticRegression(max_iter=500)
        self.svc = SVC(kernel='rbf',verbose=0)
        self.lsvc = LinearSVC(verbose=0, max_iter=2000)
        self.bnb = BernoulliNB()
        self.gnb = GaussianNB()
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000,
                                     alpha=1e-4, solver='sgd', verbose=0,
                                     tol=1e-4, random_state=1, learning_rate_init=.1)
        
        # 각 모델의 저장과 불러오기를 위한 변수명 설정
        self.lgr_modelfile = 'lgr.model'
        self.svc_modelfile = 'svc.model'
        self.lsvc_modelfile = 'lsvc.model'
        self.bnb_modelfile = 'bnb.model'
        self.gnb_modelfile = 'gnb.model'
        self.mlp_clf_modelfile = 'mlp_clf.model'
        print("[ML Model init]")

      
    def model_fit(self, x: list, y) -> None: # 훈련데이터셋 x, y를 사용하여 모델 훈련
        """
        6개의 모델 훈련
        """
        self.lgr.fit(x, y)
        self.svc.fit(x, y)
        self.lsvc.fit(x, y)
        self.bnb.fit(x, y)
        self.gnb.fit(x, y)
        self.mlp_clf.fit(x, y)
        print("[Model Fit]")


    def model_score(self, test_x: list, test_y: list, option=0) -> None: # 모델의 정확도(Accuracy) 측정
        """
        훈련된 모델의 정확도 측정.
        option = 1: 실제값과 예측값을 함께 출력
        """  
        print("Logistic Regression: ",self.lgr.score(test_x, test_y))
        print("SVC svm: ",self.svc.score(test_x, test_y)) 
        print("Linear SVC svm: ",self.lsvc.score(test_x, test_y)) 
        print("BNB: ",self.bnb.score(test_x, test_y) )
        print("GNB: ",self.gnb.score(test_x, test_y)) 
        print("Multi-layer Perceptron classifier: ",self.mlp_clf.score(test_x,test_y))
        
        if option is 1: # 실제값, 예측값 출력
            lgr_pred_y = self.lgr.predict(test_x)
            svc_pred_y = self.svc.predict(test_x)
            lsvc_pred_y = self.lsvc.predict(test_x)
            bnb_pred_y = self.bnb.predict(test_x)
            gnb_pred_y = self.gnb.predict(test_x)
            mlp_clf_pred_y = self.mlp_clf.predict(test_x)
            print('실제 Y값: ',test_y)
            print('Logistic Regression Prediction: ',list(lgr_pred_y))
            print('SVC SVM Prediction: ',list(svc_pred_y))
            print('Linear SVC SVM Prediction: ',list(lsvc_pred_y))
            print('BNB Prediction: ',list(bnb_pred_y))
            print('GNB Prediction: ',list(gnb_pred_y))
            print('MLP Prediction: ',list(mlp_clf_pred_y))
        else:
            pass
    
    def f1_score(self, test_x: list, test_y: list, option=0, graph =0) -> None: # 모델의 F1 Score 측정
        """
        훈련된 모델의 F1_score 측정.
        option = 1: 실제값과 예측값을 함께 출력.
        graph = 1: 정확도, F1_score를 그래프로 표현.
        """
        lgr_pred_y = self.lgr.predict(test_x)
        svc_pred_y = self.svc.predict(test_x)
        lsvc_pred_y = self.lsvc.predict(test_x)
        bnb_pred_y = self.bnb.predict(test_x)
        gnb_pred_y = self.gnb.predict(test_x)
        mlp_clf_pred_y = self.mlp_clf.predict(test_x)
        
        print("Logistic Regression: ", f1_score(test_y, lgr_pred_y, average='weighted'))
        print("SVC svm: ", f1_score(test_y, svc_pred_y, average='weighted'))
        print("Linear SVC svm: ", f1_score(test_y, lsvc_pred_y, average='weighted'))      
        print("bnbn: ", f1_score(test_y, bnb_pred_y, average='weighted'))
        print("gnb: ", f1_score(test_y, gnb_pred_y, average='weighted'))
        print("Multi-layer Perceptron classifier: ", f1_score(test_y, mlp_clf_pred_y, average='weighted'))
        
        if graph is 1: #정확도와 F1 Score 그래프
            bar_width=0.35
            alpha = 0.5
            x = ['lgr', 'svc','lsvc','bnb','gnb','mlp']
            y = [self.lgr.score(test_x, test_y), self.svc.score(test_x, test_y),
                 self.lsvc.score(test_x, test_y),self.bnb.score(test_x, test_y),
                 self.gnb.score(test_x, test_y),self.mlp_clf.score(test_x, test_y) ]
            y2 = [f1_score(test_y, lgr_pred_y, average='weighted'), f1_score(test_y, svc_pred_y, average='weighted'),
                  f1_score(test_y, lsvc_pred_y, average='weighted'), f1_score(test_y, bnb_pred_y, average='weighted'), 
                  f1_score(test_y, gnb_pred_y, average='weighted'), f1_score(test_y, mlp_clf_pred_y, average='weighted')]
            p1 = plt.bar(np.arange(len(x)), y, bar_width, color='b', alpha=alpha, label = 'Accuracy')
            p2 = plt.bar(np.arange(len(x))+bar_width, y2, bar_width, color = 'r', alpha=alpha, label = 'F1-score')
            plt.title('Metrics', fontsize=15)
            plt.xlabel('Algorithms', fontsize=10)
            plt.ylabel('Score', fontsize=10)
            plt.xticks(np.arange(len(x)), x, fontsize=8)
            plt.ylim(0.5, 1)
            plt.legend((p1[0],p2[0]),('Accuracy', 'F1-score'),
                       ncol=2, loc = 'lower right', fontsize=8)
            plt.show()
        
        if option is 1: # 실제값, 예측값 출력
            print('실제 Y값: ',test_y)
            print('Logistic Regression Prediction: ',list(lgr_pred_y))
            print('SVC SVM Prediction: ',list(svc_pred_y))
            print('Linear SVC SVM Prediction: ',list(lsvc_pred_y))
            print('BNB Prediction: ',list(bnb_pred_y))
            print('GNB Prediction: ',list(gnb_pred_y))
            print('MLP Prediction: ',list(mlp_clf_pred_y))
        else:
            pass

    def model_save(self) -> None:   # 인스턴스가 갖고있는 변수명으로 모델 저장
        """
        초기화된 변수 '모델명'_modelfile를 이름으로 하는 파일로 모델 저장
        다른 이름으로 저장시:
        인스턴스명.'모델명'_modelfile = '다른이름'
        인스턴스명.model_save().
        """
        dump(self.lgr, self.lgr_modelfile)
        dump(self.svc, self.svc_modelfile)
        dump(self.lsvc, self.lsvc_modelfile)
        dump(self.bnb, self.bnb_modelfile)
        dump(self.gnb, self.gnb_modelfile)
        dump(self.mlp_clf, self.mlp_clf_modelfile)
        print("[Model Save]")


    def model_load(self) -> None:   # 인스턴스가 갖고있는 변수명으로 모델 불러오기
        """
        초기화된 변수 '모델명'_modelfile를 이름으로 하는 파일로 모델 불러오기
        인스턴스명 = ML_Model()
        인스턴스명.'모델명'_modelfile = '다른이름'
        인스턴스명.model_load().
        """
        self.lgr = load(self.lgr_modelfile)
        self.svc = load(self.svc_modelfile)
        self.lsvc = load(self.lsvc_modelfile)
        self.bnb = load(self.bnb_modelfile)
        self.gnb = load(self.gnb_modelfile)
        self.mlp_clf = load(self.mlp_clf_modelfile)
        print("[Model Load]")
    
    def model_comparison(self, test_x, test_y, test_T):
        lgr_pred_y = self.lgr.predict(test_x)
        svc_pred_y = self.svc.predict(test_x)
        lsvc_pred_y = self.lsvc.predict(test_x)
        bnb_pred_y = self.bnb.predict(test_x)
        gnb_pred_y = self.gnb.predict(test_x)
        mlp_clf_pred_y = self.mlp_clf.predict(test_x)
        print('\x1b[30m'+'0: 비기술문서, 1: 기술문서')
        print('\x1b[30m'+'========Logistic_Regression========')
        for x in range(len(test_y)):
            if test_y[x] == lgr_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(lgr_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(lgr_pred_y[x]))
        print('\x1b[30m'+'===================================')
        print('\x1b[30m'+'==============SVC_SVM==============')
        for x in range(len(test_y)):
            if test_y[x] == svc_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(svc_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(svc_pred_y[x]))
        print('\x1b[30m'+'===================================')
        print('\x1b[30m'+'===========LinearSVC_SVM===========')
        for x in range(len(test_y)):
            if test_y[x] == lsvc_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(lsvc_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(lsvc_pred_y[x]))
        print('\x1b[30m'+'===================================')
        print('\x1b[30m'+'============Bernuill_NB============')
        for x in range(len(test_y)):
            if test_y[x] == bnb_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(bnb_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(bnb_pred_y[x]))
        print('\x1b[30m'+'===================================')
        print('\x1b[30m'+'============Gaussian_NB============')
        for x in range(len(test_y)):
            if test_y[x] == gnb_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(gnb_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(gnb_pred_y[x]))
        print('\x1b[30m'+'===================================')
        print('\x1b[30m'+'=========MLP_Classification========')
        for x in range(len(test_y)):
            if test_y[x] == mlp_clf_pred_y[x]:
                print('\x1b[34m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(mlp_clf_pred_y[x]))
            else:
                print('\x1b[31m'+test_T[x]+', 실제값: '+str(test_y[x])+', 예측값: '+str(mlp_clf_pred_y[x]))
        print('\x1b[30m'+'===================================')

    def __del__(self):
        print("[Model Delete]")
        


# ## 2.2 데이터처리 클래스

# ### okt 형태소 분석기 처리 예시
# <br>
# <b>okt.nouns</b>(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는')
# 
# >['항공기', '체계', '종합', '개발', '경험']
# 
# <b>okt.pos</b>(u'이것도 되나욬ㅋㅋ')
# 
# >[('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되나욬', 'Noun'), ('ㅋㅋ', 'KoreanParticle')]
# 
# <b>okt.pos</b>(u'이것도 되나욬ㅋㅋ', <b>norm=True</b>)
# 
# >[('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되나요', 'Verb'), ('ㅋㅋ', 'KoreanParticle')]
# 
# <b>okt.pos</b>(u'이것도 되나욬ㅋㅋ', <b>norm=True, stem=True</b>)
# 
# >[('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되다', 'Verb'), ('ㅋㅋ', 'KoreanParticle')]
#


class DocProcessing:
    def __init__(self):
        
        self.tech = []          # 최초 기술문서
        self.normal = []        # 최초 비기술문서
        
        self.data_set = []      # 데이터셋
        self.train_set = []     # 데이터셋을 일정 비율로 나눈 훈련용 데이터셋
        self.test_set = []      # 데이터셋을 일정 비율로 나눈 검증용 데이터셋
        
        self.pos_tagger = Okt()     # 형태소 분석기
       
        """
        각 문서내용:x, 레이블값:y, 문서제목:T
        """
        self.train_x = [] 
        self.train_y = []
        self.train_T = []
        self.test_x = []
        self.test_y = []
        self.test_T = []
        
        print("[DocProcessing init]")
        
    def print_files_in_dir(self, root_dir, docs, option=0):   # path = './labeled/기술문서/Korean/', tech
        """
        데이터를 불러오는 함수
        """
        files = os.listdir(root_dir)
        for file in files:
            path_ = os.path.join(root_dir, file)
            if os.path.isdir(path_):
                self.print_files_in_dir(path_,self, docs)
            elif path_[-13:] == '_content.json':    # content 데이터와 meta 데이터가 섞여있기 때문
                path_=path_.replace("\\","/")
                with open(path_, "r", encoding="utf-8") as json_data:
                    docs.append([path_.split('/')[-1],json.load(json_data)])
            elif path_[-12:] == '_content.txt':     # content 데이터와 meta 데이터가 섞여있기 때문
                path_=path_.replace("\\","/")
                with open(path_, "r", encoding="utf-8") as txt_data:
                    docs.append([path_.split('/')[-1],txt_data.read()])
        print("[Doc Loading]")
                    
    def labeling(self, path, option=0):     # path = './labeled/기술문서/Korean/', tech
        """
        print_files_in_dir 을 호출하여 데이터를 불러오고
        option 값에 따라 라벨링하는 함수
        """
        if option == 0:
            self.print_files_in_dir(path,self.normal)
            self.normal = [[x[0],x[1], 0] for x in tqdm(self.normal)]
            
        elif option == 1:
            self.print_files_in_dir(path, self.tech)
            self.tech = [[x[0],x[1], 1] for x in tqdm(self.tech)]
        print("[Doc labeling]")

    def preprocessing(self) -> str:
        """
        전처리 함수. 
        한글제외 모두 제거
        """
        self.data_set = copy.deepcopy(self.tech)+copy.deepcopy(self.normal)
        for x in tqdm(range(len(self.data_set))):
            review = re.sub('([^가-힣 ]+)', ' ', self.data_set[x][1])  # 한글제외 모두 제거 *******
            self.data_set[x][1] = re.sub(r' +', ' ', review)  # 많은 공백을 하나로 변환
        print("[Doc preprocessing]")
  
    def tokenize(self, option='morph'):     # 'morph': 형태소 분석 - 형태소, 'nouns': 형태소 분석 - 명사
        """
        형태소분석기 Okt를 사용하는 함수
        명사(nouns)의 경우에 한글자 단어는 제외(ex. 일, 월 등)
        """
        if option == 'morph':   # norm, stem은 optional
            self.data_set = [(row[0], ['/'.join(t) for t in self.pos_tagger.pos(row[1], norm=True, stem=True)], row[2]) 
                             for row in tqdm(self.data_set, desc='형태소분석', dynamic_ncols=False)]
        elif option == 'nouns':
            self.data_set = [(row[0], [t for t in self.pos_tagger.nouns(row[1]) if len(t)>1], row[2]) for row in
                             tqdm(self.data_set, desc='형태소분석', dynamic_ncols=False)]
        print("[Doc Tokenizing]")
    
    def data_split(self, ratio=0.7):
        """
        데이터셋을 입력받은(혹은 default)값으로 나누는 함수
        """
        random.shuffle(self.data_set)
        if ratio ==1 or ratio ==0:
            self.test_set = copy.deepcopy(self.data_set)
            self.test_x = [x[1] for x in self.test_set]
            self.test_y = [x[2] for x in self.test_set]
            self.test_T = [x[0] for x in self.test_set]
            print("[Doc split]")
        else:
            self.train_set = copy.deepcopy(self.data_set[:int(len(self.data_set)*ratio)])
            self.test_set = copy.deepcopy(self.data_set[int(len(self.data_set)*ratio):])

            self.train_x = [x[1] for x in self.train_set]
            self.train_y = [x[2] for x in self.train_set]
            self.train_T = [x[0] for x in self.train_set]

            self.test_x = [x[1] for x in self.test_set]
            self.test_y = [x[2] for x in self.test_set]
            self.test_T = [x[0] for x in self.test_set]
            print("[Doc split]")
    
    def __del__(self):
        print("Data Processing Class Delete")


# ## 2.3 임베딩 모델 클래스

class Embedding:
    def __init__(self):
        """
        임데딩모델 클래스 초기화
        임베딩모델 학습에 사용하는 파라미터, 생성자, 저장명 등을 초기화
        """
        self.modelfile = "doc2vec.model"
        self.cores = multiprocessing.cpu_count()    # cpu 코어 개수
        self.vector_size = 300                      # 문서의 벡터 차원(개수)
        self.window_size = 15                       # 훈련에 사용되는 현재와 예측 단어의 최대거리
        self.word_min_count = 5                     # 훈련사용할 단어의 빈도 기준
        self.sampling_threshold = 1e-5              # 고빈도 단어의 다운 샘플링 임계값
        self.negative_size = 5                      # 네거티브 샘플링 개수
        self.train_epoch = 100                      # iteration 횟수
        self.dm = 1                                 # 0 = dbow; 1 = dmpv, 임베딩 모델의 훈련 알고리즘 선택
        self.worker_count = self.cores              # number of parallel processes

        self.doc_vectorizer = doc2vec.Doc2Vec(min_count=self.word_min_count, window=self.window_size,
                                              vector_size=self.vector_size, alpha=0.025, min_alpha=0.025,
                                              seed=1234, workers=self.worker_count, epochs=self.train_epoch,
                                              dm=self.dm, negative=self.negative_size,
                                              sample=self.sampling_threshold)
        
        print("[Embedding Model Init]")
        
    def em_train(self, corpus):
        """
        입력받은 데이터를 사용하여 모델 학습
        모델의 형태는 리스트. [[제목, 내용, 레이블], ...]
        """
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')
        tagged_docs = [TaggedDocument(d, [c]) for e, d, c in corpus]
        
        self.doc_vectorizer.build_vocab(tagged_docs)
        self.doc_vectorizer.train(tagged_docs, epochs=self.doc_vectorizer.epochs,
                                  total_examples=self.doc_vectorizer.corpus_count)
        print("[Embedding Model Train]")

    def em_run(self, docs):
        """
        학습된 모델을 사용해 입력받은 문서를 임베딩하는 함수
        """
        print("[Embedding ...]")
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')
        tagged_docs = [TaggedDocument(d, [c]) for e, d, c in docs]
        return [self.doc_vectorizer.infer_vector(doc.words) for doc in tqdm(tagged_docs)]

    def em_save(self):
        """
        학습된 모델을 저장
        """
        self.doc_vectorizer.save(self.modelfile)
        print("[Embedding Model Save]")

    def em_load(self):
        """
        학습된 모델을 불러오기
        """
        self.doc_vectorizer = doc2vec.Doc2Vec.load(self.modelfile)
        print("[Embedding Model Load]")
    
    def __del__(self):
        """
        소멸자
        """
        print("[Embedding Model Delete]")


if __name__ == "__main__":
    print(__name__)

    # # 3. 코드 실행
    doc = DocProcessing()

    doc.labeling('./labeled/비기술문서/Korean/', option=0)
    doc.labeling('./labeled/기술문서/Korean/', option=1)

    doc.preprocessing()

    doc.tokenize(option='nouns')

    doc.data_split()

    em = Embedding()
    em.em_train(doc.train_set)

    doc.train_x = em.em_run(doc.train_set)
    doc.test_x = em.em_run(doc.test_set)

    """
    모델 생성 및 학습
    """
    ml = ML_Model()
    ml.model_fit(doc.train_x, doc.train_y)

    """
    생성된 모델의 정확도(Accuracy) 측정, 실제값과 예측값 확인
    """
    ml.model_score(doc.test_x, doc.test_y, option=0)  # option = 0 or 1

    """
    생성된 모델의 F1 score 측정, 실제값과 예측값 확인, 그래프 그리기
    """
    ml.f1_score(doc.test_x, doc.test_y, option=0, graph=1)  # option = 0 or 1, graph = 0 or 1

    ml.model_comparison(doc.test_x[:20], doc.test_y[:20], doc.test_T[:20])

    """
    생성된 모델의 저장
    """
    ml.model_save()

    """
    생성된 모델 불러오기
    """
    ml.model_load()

    # # 기타
    # 메모리 해제
    del doc
    del em
    del ml

    print("Delete all instance")
