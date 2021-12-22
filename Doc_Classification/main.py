# -*- coding:utf-8 -*-
# created by 데이터 응용개발팀 이지형

from Doc_Classification import Doc_Classification

if __name__ == '__main__':

    doc = Doc_Classification.DocProcessing()

    doc.labeling('./labeled/비기술문서/Korean/', option=0)
    doc.labeling('./labeled/기술문서/Korean/', option=1)

    doc.preprocessing()
    doc.tokenize(option='nouns')
    doc.data_split()

    em = Doc_Classification.Embedding()
    #em.em_load()
    em.em_train(doc.train_set)

    doc.train_x = em.em_run(doc.train_set)
    doc.test_x = em.em_run(doc.test_set)

    """
    모델 생성 및 학습
    """
    ml = Doc_Classification.ML_Model()
    #ml.model_load()
    ml.model_fit(doc.train_x, doc.train_y)
    #
    # """
    # 생성된 모델의 정확도(Accuracy) 측정, 실제값과 예측값 확인
    # """
    ml.model_score(doc.test_x, doc.test_y, option=0)  # option = 0 or 1
    # """
    # 생성된 모델의 F1 score 측정, 실제값과 예측값 확인, 그래프 그리기
    # """
    ml.f1_score(doc.test_x, doc.test_y, option=0, graph=1)  # option = 0 or 1, graph = 0 or 1
    #
    # """
    # 생성된 모델의 실제 결과 비교
    # """
    ml.model_comparison(doc.test_x[:10], doc.test_y[:10], doc.test_T[:10])
    #ml.model_save()

    doc_test = Doc_Classification.DocProcessing()
    doc_test.labeling('./Test/비기술/',option=0)
    doc_test.labeling('./Test/기술/',option=1)

    doc_test.preprocessing()
    doc_test.tokenize(option='nouns')
    doc_test.data_split(ratio=0)

    doc_test.test_x = em.em_run(doc_test.test_set)
    ml.model_score(doc_test.test_x,doc_test.test_y)
    ml.f1_score(doc_test.test_x, doc_test.test_y, option=0, graph=1)
    ml.model_comparison(doc_test.test_x, doc_test.test_y, doc_test.test_T)
    print("Done")

