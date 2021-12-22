# -*- coding: utf-8 -*-
"""
CHAT BOT Using Seq2Seq(Encoder and Decoder)
"""

# konlpy, mecab 등 설치
# !pip3 install --upgrade pip
# !pip3 install konlpy
# !pip3 install tweepy==3.10.0

# 라이브러리 임포트
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from konlpy.tag import Okt

# 경로와 상수
DATA_PATH = './data/'
RESULT_PATH = './result/'

DRIVE_VOCAB_PATH_WELLNESS = 'vocab_wellness.txt'
TRAIN_INPUTS_WELLNESS = 'train_inputs_wellness.npy'
TRAIN_OUTPUTS_WELLNESS = 'train_outputs_wellness.npy'
TRAIN_TARGETS_WELLNESS = 'train_targets_wellness.npy'
DATA_CONFIGS_WELLNESS = 'data_configs_wellness.json'

DRIVE_VOCAB_PATH_CAHTBOT = 'vocab_chatbot.txt'
TRAIN_INPUTS_CHATBOT = 'train_inputs_chatbot.npy'
TRAIN_OUTPUTS_CHATBOT = 'train_outputs_chatbot.npy'
TRAIN_TARGETS_CHATBOT = 'train_targets_chatbot.npy'
DATA_CONFIGS_CHATBOT = 'data_configs_chatbot.json'

MODEL_NAME = 'seq2seq'
BATCH_SIZE = 2
MAX_SEQUENCE = 20
EPOCH = 30
UNITS = 1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1

# 챗봇 데이터
data_chatbot = pd.read_csv(DATA_PATH + 'ChatbotData.csv', encoding='utf-8')
data_wellness = pd.read_excel(DATA_PATH + '웰니스_대화_스크립트_데이터셋.xlsx')

data_chatbot.dropna(axis=0, inplace=True)
data_wellness.dropna(axis=0, inplace=True)

question_chatbot, answer_chatbot = list(data_chatbot['Q']), list(data_chatbot['A'])
question_wellness, answer_wellness = list(data_wellness['유저']), list(data_wellness['챗봇'])


# 전처리 일종 - 영어 한글 숫자 제외 삭제
def data_preprocess(data):
    words = []
    for x in data:
        x = re.sub('^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9', " ", str(x)).split()
        for y in x:
            words.append(y)
    return [word for word in words if word]


# 형태소 분석
def morph(text):
    okt = Okt()
    result = []
    for x in text:
        temp = " ".join(okt.morphs(str(x).replace(' ', '')))
        result.append(temp)
    return result


# 단어사전 구축 및 저장
def make_vocab(path, question, answer):
    data = []

    # question = morph(question)
    # answer = morph(answer)

    data.extend(question)
    data.extend(answer)
    data = data_preprocess(data)
    data = list(set(data))

    # 각 패딩, 시작, 끝, 모르는 단어 토큰
    data[:0] = ["<PAD>", "<CLS>", "<SEP>", "<UNK>"]
    with open(path, 'w', encoding='UTF-8') as vocab:
        for x in data:
            vocab.write(x + '\n')
    return data


# 구축한 단어 사전 열기
def open_vocab(path):
    result = []
    with open(path, 'r', encoding='utf-8') as vocab:
        for x in vocab:
            result.append(x.strip())
    return result


# wordidx, idxword 구분해서 저장하고 vocab_size 구하기
def vo(data):
    wordidx = {word: idx for idx, word in enumerate(data)}
    idxword = {idx: word for idx, word in enumerate(data)}
    vocab_size = len(wordidx)
    return wordidx, idxword, vocab_size


# 인코더에 넣는 단어를 임베딩, 단어를 사전의 idx로 바꾸고 없으면 <UNK> 토큰으로 교체
# MAX_LEN에 따라 패딩작업 진행
def encoding_data(text, vocab_dic, MAX_LEN=20):
    input_data = []
    texts_len = []
    # text = morph(text)
    for x in text:
        x = re.sub('^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9', "", str(x)).split()
        temp = []
        for y in x:
            if vocab_dic.get(y) is not None:
                temp.extend([vocab_dic[y]])
            else:
                temp.extend([vocab_dic['<UNK>']])

        if len(temp) > MAX_LEN:
            temp = temp[:MAX_LEN]

        texts_len.append(len(temp))
        temp = temp + (MAX_LEN - len(temp)) * [vocab_dic['<PAD>']]

        input_data.append(temp)

    return np.asarray(input_data), texts_len


# 디코더에 넣는 단어를 임베딩, 단어를 사전의 idx로 바꾸고 없으면 <UNK> 토큰으로 교체
# 맨 앞에 시작 토큰 CLS 삽입, MAX_LEN에 따라 패딩작업 진행
def decoding_input(text, vocab_dic, MAX_LEN=20):
    input_data = []
    texts_len = []
    # text = morph(text)
    for x in text:
        x = re.sub('^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9', "", str(x)).split()
        temp = []
        temp = [vocab_dic['<CLS>']] + [vocab_dic[y] if y in vocab_dic else vocab_dic['<UNK>'] for y in x]
        if len(temp) > MAX_LEN:
            temp = temp[:MAX_LEN]

        texts_len.append(len(temp))
        temp = temp + (MAX_LEN - len(temp)) * [vocab_dic['<PAD>']]

        input_data.append(temp)

    return np.asarray(input_data), texts_len


# 훈련에 사용할 디코더 결과값을 임베딩
# 마지막에 SEP 토큰 삽입 후 패딩
def decoding_result(text, vocab_dic, MAX_LEN=20):
    input_data = []
    # text = morph(text)
    for x in text:
        x = re.sub('^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9', "", str(x)).split()
        temp = []
        temp = [vocab_dic[y] if y in vocab_dic else vocab_dic['<UNK>'] for y in x]
        if len(temp) >= MAX_LEN:
            temp = temp[:MAX_LEN - 1] + [vocab_dic['<SEP>']]
        else:
            temp += [vocab_dic['<SEP>']]

        temp += (MAX_LEN - len(temp)) * [vocab_dic['<PAD>']]
        input_data.append(temp)

    return np.asarray(input_data)


# plot 그리는 함수
def plot_graphs(history, metrics):
    plt.plot(history.history[metrics])
    plt.plot(history.history['val_' + metrics], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics)
    plt.legend([metrics, 'val_' + metrics])
    plt.show()


# 인코더
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dimension_num, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.dimension_num = dimension_num
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.dimension_num)
        # LSTM, GRU 등 RNN Cell
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # x는 인풋, hidden의 경우 initialize_hidden_state값
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # 초기 히든값
    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0], self.enc_units))


# BahdanauAttention, tf.contrib.seq2seq.BahdanauAttention(num_units=hp.attention_depth, memory=enc_outs)
# 어텐션 알고리즘, 인코더의 hidden state 벡터를 받아 연산을 수행후 적절히 디코더 셀에 반환
# 컨텍스트 벡터와 어텐션 가중치를 반환
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# 디코더
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dimension_num, dec_units, batch_sz):
        super(Decoder, self).__init__()

        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.dimension_num = dimension_num

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.dimension_num)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        # cell의 결과값
        x = self.fc(output)

        return x, state, attention_weights


# vocab_size: 사전크기, dimension_num: 차원 크기, UNITS: LSTM cell 결과 차원, 어텐션 차원,
# BATCH_SIZE: 배치사이즈, wordidx_chatbot: word-idx사전
class Seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, dimension_num, enc_units, dec_units, batch_sz, wordidx, end_token_idx=2):
        super(Seq2seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, dimension_num, enc_units, batch_sz)
        self.decoder = Decoder(vocab_size, dimension_num, dec_units, batch_sz)
        self.wordidx = wordidx

    def call(self, x):

        inp, tar = x
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        predict_tokens = list()

        for t in range(0, tar.shape[1]):
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32)
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))
        return tf.stack(predict_tokens, axis=1)

    # 추론
    def inference(self, x):
        inp = x

        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.wordidx["<CLS>"]], 1)

        predict_tokens = list()

        for t in range(0, MAX_SEQUENCE):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])

            if predict_token == self.end_token_idx:
                break

            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)

        return tf.stack(predict_tokens, axis=0).numpy()


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')


# 손실함수 정확도함수 선언
def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask
    acc = train_accuracy(real, pred)
    return tf.reduce_mean(acc)


# 콜백 함수
checkpoint_path_chatbot = '/weights_chatbot.h5'
cp_callback_chatbot = ModelCheckpoint(
    checkpoint_path_chatbot, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

checkpoint_path_wellness = '/weights_wellness.h5'
cp_callback_wellness = ModelCheckpoint(
    checkpoint_path_wellness, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)


# 단어사전 구축, wordidx, idxword, 입력-출력 데이터 전처리 - 일반 챗봇데이터
chat_bot = make_vocab(DRIVE_VOCAB_PATH_CAHTBOT, question_chatbot, answer_chatbot)
wordidx_chatbot, idxword_chatbot, vocab_size_chatbot = vo(chat_bot)
input_data_chatbot, texts_len_chatbot = encoding_data(question_chatbot, wordidx_chatbot)
output_data_chatbot, out_texts_len_chatbot = decoding_input(answer_chatbot, wordidx_chatbot)
target_data_chatbot = decoding_result(answer_chatbot, wordidx_chatbot)
data_configs_chatbot = {'wordidx':wordidx_chatbot, 'idxword':idxword_chatbot,
                        'vocab_size':vocab_size_chatbot, 'PAD': "<PAD>",
                        'CLS': "<CLS>", 'SEP':"<SEP>", 'UNK':"<UNK>"}

# 데이터 저장 - 일반 챗봇데이터
np.save(open(DATA_PATH + TRAIN_INPUTS_CHATBOT, 'wb'), input_data_chatbot)
np.save(open(DATA_PATH + TRAIN_OUTPUTS_CHATBOT, 'wb'), output_data_chatbot)
np.save(open(DATA_PATH + TRAIN_TARGETS_CHATBOT, 'wb'), target_data_chatbot)
json.dump(data_configs_chatbot, open(DATA_PATH + DATA_CONFIGS_CHATBOT, 'w'))

# 데이터 불러오기 - 일반 챗봇데이터
index_inputs_chatbot = np.load(open(DATA_PATH + TRAIN_INPUTS_CHATBOT, 'rb'))
index_outputs_chatbot = np.load(open(DATA_PATH + TRAIN_OUTPUTS_CHATBOT , 'rb'))
index_targets_chatbot = np.load(open(DATA_PATH + TRAIN_TARGETS_CHATBOT , 'rb'))
prepro_configs_chatbot = json.load(open(DATA_PATH + DATA_CONFIGS_CHATBOT, 'r'))

# 입출력 데이터를 제외하고는 dictionary-json 형식으로 불러오기 - 일반 챗봇데이터
wordidx_chatbot = prepro_configs_chatbot['wordidx']
idxword_chatbot = prepro_configs_chatbot['idxword']
std_index_chatbot = prepro_configs_chatbot['CLS']
end_index_chatbot = prepro_configs_chatbot['SEP']
vocab_size_chatbot = prepro_configs_chatbot['vocab_size']

# 단어사전 구축, wordidx, idxword, 입력-출력 데이터 전처리
wellness = make_vocab(DRIVE_VOCAB_PATH_WELLNESS, question_wellness, answer_wellness)
wordidx_wellness, idxword_wellness, vocab_size_wellness = vo(wellness)
input_data_wellness, texts_len_wellness = encoding_data(question_wellness, wordidx_wellness)
output_data_wellness, out_texts_len_wellness = decoding_input(answer_wellness, wordidx_wellness)
target_data_wellness = decoding_result(answer_wellness, wordidx_wellness)
data_configs_wellness = {'wordidx':wordidx_wellness, 'idxword':idxword_wellness,
                        'vocab_size':vocab_size_wellness, 'PAD': "<PAD>",
                        'CLS': "<CLS>", 'SEP':"<SEP>", 'UNK':"<UNK>"}
# 데이터 저장 - wellness 챗봇데이터
np.save(open(DATA_PATH + TRAIN_INPUTS_WELLNESS, 'wb'), input_data_wellness)
np.save(open(DATA_PATH + TRAIN_OUTPUTS_WELLNESS, 'wb'), output_data_wellness)
np.save(open(DATA_PATH + TRAIN_TARGETS_WELLNESS, 'wb'), target_data_wellness)
json.dump(data_configs_wellness, open(DATA_PATH + DATA_CONFIGS_WELLNESS, 'w'))

# 데이터 불러오기 - wellness 챗봇데이터
index_inputs_wellness = np.load(open(DATA_PATH + TRAIN_INPUTS_WELLNESS, 'rb'))
index_outputs_wellness = np.load(open(DATA_PATH + TRAIN_OUTPUTS_WELLNESS , 'rb'))
index_targets_wellness = np.load(open(DATA_PATH + TRAIN_TARGETS_WELLNESS, 'rb'))
prepro_configs_wellness = json.load(open(DATA_PATH + DATA_CONFIGS_WELLNESS, 'r'))

# 입출력 데이터를 제외하고는 dictionary-json 형식으로 불러오기 - wellness 챗봇데이터
wordidx_wellness = prepro_configs_wellness['wordidx']
idxword_wellness = prepro_configs_wellness['idxword']
std_index_wellness = prepro_configs_wellness['CLS']
end_index_wellness = prepro_configs_wellness['SEP']
vocab_size_wellness = prepro_configs_wellness['vocab_size']

# Seq2Seq 모델 생성- 일반 챗봇데이터
model_chatbot = Seq2seq(vocab_size_chatbot, EMBEDDING_DIM, UNITS, UNITS,
                        BATCH_SIZE, wordidx_chatbot,wordidx_chatbot["<SEP>"])

# Seq2Seq 모델 컴파일- 일반 챗봇데이터
model_chatbot.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[accuracy])

# Seq2Seq 모델 훈련- 일반 챗봇데이터
history_chatbot = model_chatbot.fit(
    [index_inputs_chatbot, index_outputs_chatbot], index_targets_chatbot,
    batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=VALIDATION_SPLIT,
    callbacks=[earlystop_callback, cp_callback_chatbot]
)

# Seq2Seq 모델 생성 - wellness 챗봇데이터
model_wellness = Seq2seq(vocab_size_wellness, EMBEDDING_DIM, UNITS, UNITS,
                         BATCH_SIZE, wordidx_wellness,wordidx_wellness["<SEP>"])

# Seq2Seq 모델 컴파일 - wellness 챗봇데이터
model_wellness.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[accuracy])

# Seq2Seq 모델 훈련 - wellness 챗봇데이터
history_wellness = model_wellness.fit(
    [index_inputs_wellness, index_outputs_wellness], index_targets_wellness,
    batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=VALIDATION_SPLIT,
    callbacks=[earlystop_callback, cp_callback_wellness]
)

# 모델 요약 및 차트 출력
model_chatbot.summary()
model_wellness.summary()

plot_graphs(history_chatbot, 'accuracy')
plot_graphs(history_chatbot, 'loss')

plot_graphs(history_wellness, 'accuracy')
plot_graphs(history_wellness, 'loss')

# 결과 예시
ques = "sns 중독이 심해"
test_index_inputs, _ = encoding_data([ques], wordidx_chatbot)
predict_tokens = model_chatbot.inference(test_index_inputs)
print(predict_tokens)
print(' '.join([idxword_chatbot[str(t)] for t in predict_tokens]))

ques = "남편이 무서워"
test_index_inputs, _ = encoding_data([ques], wordidx_wellness)
print(test_index_inputs)
predict_tokens = model_wellness.inference(test_index_inputs)
print(predict_tokens)
print(' '.join([idxword_wellness[str(t)] for t in predict_tokens]))
