
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint  # , EarlyStopping


def scaled_dot_product_attention(query, key, value, mask):  # 1번
    # d_model: 출력차원 크기
    # num_heads: 어텐셜을 병렬로 수행할 때의 병렬 수행 갯수
    # query shape : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key shape : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value shape : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # 인코더(k, v)-디코더(q) 에서는 query 길이와 key, value의 길이는 다를 수 있음
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)
    # mask: (..., seq_len_q, seq_len_k). 기본값은 None.

    # query와 key의 곱. 어텐션 스코어 맵. 2번
    # transpose_b는 key의 전치를 위한 옵션
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링 2-1번
    # Dk의 루트값으로 나눠준다.
    # Dk: key 벡터의 차원 수
    depth = tf.cast(tf.shape(key)[-1], tf.float32)  # Dk
    logits = matmul_qk / tf.math.sqrt(depth)

    # softmax함수를 사용해 확률값으로 변환, 4번
    # 소프트맥스는 key의 문장 길이 방향으로 진행
    # attention weight : (batch_size, num_heads, query의 문장 길이, 'key의 문장 길이')
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # 모든 value(모든 토큰)에 곱하고 그 값들을 더하면 해당 query에 대한 문맥벡터가 나옴, 5번
    output = tf.matmul(attention_weights, value)

    return output, attention_weights  # attention_weight도 출력하는 이유는 디코더를 참고


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.projection_dim = d_model // num_heads
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, d_model]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, d_model)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        return outputs


# 포지션 와이즈 피드 포워드 네트워크
def position_wise_feed_forward_network(dff, d_model):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class classification_encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(classification_encoder, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)  # 멀티 헤드 어텐션
        self.ffn = position_wise_feed_forward_network(dff, d_model)  # 포지션 와이즈 피드 포워드 신경망
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 레이어 노말라이제이션
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 레이어 노말라이제이션
        self.dropout1 = tf.keras.layers.Dropout(rate)  # 드롭아웃
        self.dropout2 = tf.keras.layers.Dropout(rate)  # 드롭아웃

    def call(self, inputs, training):
        attn_output = self.att(inputs)  # 멀티 헤드 어텐션
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        ffn_output = self.ffn(out1)  # 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm


# 임베딩에 위치정보 자체를 대입
# BERT에서 사용한다고 알려짐
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, d_model):
        super(PositionEmbedding, self).__init__()
        # 토큰 임베딩
        self.token_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        # 포지션 임베딩
        self.pos_emb = tf.keras.layers.Embedding(max_len, d_model)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        # 0부터 시퀀스 최대길이 만큼의 값을 임베딩
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        # 토큰값 임베딩
        x = self.token_emb(x)
        return x + positions


def tokenize_and_filter_one(inputs, MAX_LENGTH, tokenizer):
    tokenized_inputs = []

    for sentence1 in inputs:
        # print(sentence1)
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN_ + tokenizer.encode(sentence1) + END_TOKEN_
        # print(sentence1)
        tokenized_inputs.append(sentence1)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    return tokenized_inputs


DATA_PATH = './data_in/'

data = pd.read_csv(DATA_PATH + "F01.v_211122_newsData_before2.3.textPreprocessing.csv", encoding="UTF-8")


tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(data['contents'], target_vocab_size=2 ** 13)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# -> 이 경우 token의 양이 많고 시작, 종료와 같은 특수토큰이 이미 존재함
# 다만 한글의 경우 의미있게 잘린다고 판단을 내리기 힘듦, ex) 안녕하세요-> 안_, *녕, *하, *세, *요

START_TOKEN_, END_TOKEN_ = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
max_len = 40  # 문장의 최대 길이
d_model = 32  # 각 단어의 임베딩 벡터의 차원
num_heads = 2  # 어텐션 헤드의 수
dff = 32  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기

questions_ = [tokenize_and_filter_one([x], max_len, tokenizer) for x in tqdm(data['contents'])]
questions = np.asarray(questions_).reshape(-1, 40)
answers = data['category']

X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.33)

y_train_categorical = tf.keras.utils.to_categorical(y_train)
y_test_categorical = tf.keras.utils.to_categorical(y_test)


inputs = tf.keras.layers.Input(shape=(max_len,))
# embedding_layer = PositionalEncoding(vocab_size, d_model)
embedding_layer = PositionEmbedding(max_len, VOCAB_SIZE, d_model)

# ##########################################################
x = embedding_layer(inputs)
transformer_block = classification_encoder(d_model, num_heads, dff)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(20, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# ##########################################################

checkpoint_path_transformer = DATA_PATH + '/weights_transformer_classification.h5'
MONITOR = 'val_accuracy'  # 'accuracy'
cp_callback_transformer = ModelCheckpoint(
    checkpoint_path_transformer, monitor=MONITOR, verbose=1, save_best_only=True, save_weights_only=True)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test),
                    callbacks=cp_callback_transformer)

print("테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
# ##################################################
temp = np.asarray(
    tokenize_and_filter_one([
        "Jets Chairman Christopher Johnson Won't Fine Players For Anthem Protests."
    ], max_len, tokenizer)).reshape(-1, 40)  # 6

y_pred = model.predict(temp)
result = np.argmax(y_pred, axis=-1)
result
# ##################################################
temp = np.asarray(
    tokenize_and_filter_one([
        'Premier League roundup: Manchester City extends lead at the top as Chelsea loses ground in title race'
    ], max_len, tokenizer)).reshape(-1, 40)  # 6

y_pred = model.predict(temp)
result = np.argmax(y_pred, axis=-1)
result
# ##################################################

