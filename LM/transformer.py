# -*- coding: utf-8 -*-

# !pip install tensorflow==2.6.0
# !pip install transformers
# !pip install keras~=2.6.0
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint  # , EarlyStopping


# 포지셔널 인코딩
# RNN 과 달리 입력값으로 단어를 순차적으로 넣지 않고 문장 시퀀스를 통째로
# 행렬화하여 넣는 트랜스포머는 시퀀스의 순서에 대한 정보를 주입할 필요가 있음
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        """
    d_model: 출력차원의 크기
    """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
    위치정보값을 만들기 위해 사용하는 함수
    position/10000^(2i/d_model)
    position: 인덱스 위치 리스트
    i: 차원 리스트
    d_model: 출력차원 크기
    """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 배열의 짝수 인덱스에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # 0으로 초기화된 행렬을 만들고
        angle_rads = np.zeros(angle_rads.shape)
        # 짝수에는 sine, 홀수에는 cosine함수를 적용한 값들을 저장
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        # 새로운 차원
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


"""멀티헤드 어텐션
: 내적 어텐션 구조가 중첩된 형태
-> 내적 어텐션, 순방향 어텐션 마스크

내적 어텐션 (Dot product attention)

입력: query, key, value

query: 찾는 값

key: 찾은 값

value: 의미

1. 시퀀스의 특정 토큰이 query가 되고, 시퀀스의 나머지 모든 토큰이 key가 됨

2. 이때 특정 query(특정 토큰)와 모든 key(모든 토큰)을 내적해 어텐션 스코어를 구함

  2-1.  scaling

3. 옵션으로  mask를 적용

4. softmax함수를 사용해 확률값으로 변환

5. 모든 value(모든 토큰)에 곱하고 그 값들을 더하면 해당 query에 대한 문맥벡터가 나옴

6. 2에서 모든 query와 key간의 어텐션 스코어는 Matmul같은 행렬곱으로 일괄처리가 가능
-------------------------------
스케일 내적 어텐션 (scaled dot product attention)

내적어텐션과 동일하지만 중간에 크기를 조정하는 과정(scaling)이 포함됨

2번의 어텐션스코어에 scaling 함수를 적용

--------------------------------
순방향 마스크 어텐션(Subsequent masked attention)

포지셔널 인코딩과 비슷한 상황

트랜스포머는 시퀀스를 한 번에 적용하기 때문에 예측에 있어서 직관적으로 올바르지 못한 상황과 맞닥뜨림

아직 예측하지도 않았는데 미래단어를 입력값으로 사용하는 상황이 벌어질 수 있음

따라서 자신보다 뒤에있는 단어를 마스킹해 예측에 사용하지 않는 방법을 순방향 마스크 어텐션이라 함

2에서 구한 어텐션스코어로 행렬(어텐션맵)을 생성한뒤 query뒤의 key와 내적해 구한 어텐션 값들을 마스킹하는 것

예를 들어 왼쪽 상단 - 오른쪽 하단 대각선 기준으로 상단부분을 마스킹하면 완료

일반적으로 마스킹은 옵션이며 마스킹 값으로는 -1e9와 같은 매우 작은 값을 넣어 다음 단계인 4단계 softmax함수에서 0에 수렴하는 값을 얻도록 함
"""


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

    # 마스킹. 3번, 옵션, 디코더에서 주로
    # -1e9와 같은 매우 작은 값을 넣어 다음 단계인 4단계 softmax함수에서 0에 수렴하는 값을 얻도록 함
    if mask is not None:
        logits += (mask * -1e9)

    # softmax함수를 사용해 확률값으로 변환, 4번
    # 소프트맥스는 key의 문장 길이 방향으로 진행
    # attention weight : (batch_size, num_heads, query의 문장 길이, 'key의 문장 길이')
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # 모든 value(모든 토큰)에 곱하고 그 값들을 더하면 해당 query에 대한 문맥벡터가 나옴, 5번
    output = tf.matmul(attention_weights, value)

    return output, attention_weights  # attention_weight도 출력하는 이유는 디코더를 참고


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        if self.d_model % self.num_heads != 0:
            raise Exception("self.d_model % self.num_heads != 0")

        # d_model을 num_heads로 나눈 값으로 각 병렬처리에 입력될 벡터의 차원 수
        # d_model: 512, num_heads: 8, depth: 64
        self.depth = d_model // self.num_heads

        # query, key, value의 벡터를 얻게 하기 위한 밀집층 (Wq, Wk, Wv)
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # 결과를 출력하기 위한 밀집층 (W0)
        self.dense = tf.keras.layers.Dense(units=d_model)

    # query_dense등을 통해 구한 q, k, v벡터를 num_heads수 만큼 분리
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 스케일드 닷 프로덕트 어텐션을 끝마친 값들을 다시 합침
    def concat_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, inputs):
        # 각 쿼리, 키, 벨류, 마스크 선언
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. 내적연산을 위한 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. split_heads를 사용한 나누기
        # q : (batch_size, num_heads, query의 문장 길이, depth)
        # k : (batch_size, num_heads, key의 문장 길이, depth)
        # v : (batch_size, num_heads, value의 문장 길이, depth)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # 4. 헤드값을 split된 결과들을 연결(concatenate)하기
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        # -> (batch_size, query의 문장 길이, d_model)
        concat_attention = self.concat_heads(scaled_attention, batch_size)

        # 5. 결과를 출력하기 위한 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs  # , attention_weights


# 패딩을 하기위한 추가행렬
# 0 -> 1, 0이 아닌 수 -> 0
def create_padding_mask(x):
    padding_mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


# 포지션 와이즈 피드 포워드 네트워크
def position_wise_feed_forward_network(dff, d_model):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# 하나의 인코더 레이어 -> 첫논문에서는 6개
def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    # 인풋값
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션
    # attention = MultiHeadAttention(d_model, num_heads)
    attention = MultiHeadAttention(d_model, num_heads)({
        'query': inputs, 'key': inputs, 'value': inputs,  # Q = K = V
        'mask': padding_mask  # 패딩 마스크 사용
    })

    # 드롭아웃 + 레지듀얼 커넥션과 레이어 노말라이제이션
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망
    outputs = position_wise_feed_forward_network(dff, d_model)(attention)

    # 드롭아웃 + 레지듀얼 커넥션과 레이어 노말라이제이션
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


# 레이어를 쌓는 인코더
def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    # dff: 은닉층의 크기

    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 워드 임베딩
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # 워드 임베딩값에 대한 스케일링
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 인코더를 num_layers개 쌓기 -> 논문에서는 6개
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name="encoder_layer_{}".format(i), )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


"""인코더에서는 내적 어텐션을 하는 경우에 패딩이 필요했지만 디코더는 입력부터 패딩이 필요하다

seq2seq 구조의 decoder 입력값에 대해 살펴볼것

인코더의 셀프 어텐션 : padding mask를 전달

디코더의 첫번째 서브층인 마스크드 셀프 어텐션 : look_ahead_mask, padding mask를 전달

디코더의 두번째 서브층인 인코더-디코더 어텐션 : padding mask를 전달
"""


# 디코더에서 미래값을 Masking하기 위한 함수 - 마스크드 멀티 헤드 어텐션에서 진행
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)  # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)


"""디코더의 구조는 인코더와 매우 흡사함

인코더 구조 앞에 마스크드 멀티 헤드 어텐션이 추가된 형태

두번째 멀티 헤드 어텐션에 인코더의 결과값을 추가하는 것을 잊지 말 것

query: 마스크드 멀티헤드 어텐션의 값

key, value: 인코더의 결과값
"""


def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    # 인풋값
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    # 인코더의 결과값, key, value
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # 룩어헤드 마스크 + 패딩 마스크
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")

    # 멀티-헤드 어텐션
    attention1 = MultiHeadAttention(d_model, num_heads)(
        inputs={'query': inputs, 'key': inputs, 'value': inputs,  # query = key = value
                'mask': look_ahead_mask  # 룩어헤드 마스크 + 패딩 마스크
                })

    # 레지듀얼 커넥션과 레이어 노말라이제이션
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # 패딩 마스크
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 멀티-헤드 어텐션
    attention2 = MultiHeadAttention(d_model, num_heads)(
        inputs={'query': attention1, 'key': enc_outputs, 'value': enc_outputs,  # query != key = value
                'mask': padding_mask  # 패딩 마스크
                })

    # 드롭아웃 + 레지듀얼 커넥션과 레이어 노말라이제이션
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망
    outputs = position_wise_feed_forward_network(dff, d_model)(attention2)

    # 드롭아웃 + 레지듀얼 커넥션과 레이어 노말라이제이션
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    # 인풋값
    inputs = tf.keras.Input(shape=(None,), name='inputs')

    # 인코더의 결과값, key, value
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    # 룩어헤드 마스크 + 패딩 마스크
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    # 패딩 마스크
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 워드 임베딩
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # 워드 임베딩값에 대한 스케일링
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 디코더를 num_layers개 쌓기 -> 논문에서는 6개
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name='decoder_layer_{}'.format(i), )(
            inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):
    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(
        inputs)

    # 디코더의 룩어헤드 마스크
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None),
                                             name='look_ahead_mask')(dec_inputs)

    # 디코더의 패딩 마스크
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(
        inputs)

    # 인코더의 출력
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout, )(
        inputs=[inputs, enc_padding_mask])  # 인코더의 입력은 인코더 입력과 패딩 마스크

    # 디코더의 출력
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout, )(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    # 디코더의 입력은 디코더 입력, 인코더 결과, 룩어헤드, 패딩 마스크

    # 최종 출력층
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


"""vocab_size = 9000 -> 단어집합 크기

num_layers = 4 -> 인코더, 디코더 레이어의 갯수

dff = 512 -> 포지션 와이즈 피드 포워드 신경망의 은닉층 갯수

d_model = 128 -> 입, 출력의 차원

num_heads = 4 -> 멀티-헤드 어텐션에서 병렬적으로 사용할 헤드의 수

dropout = 0.3 -> 드롭아웃률
"""


# 다중클래스 분류, 크로스 엔트로피 함수 사용
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


# 텐서플로우 dataset을 이용하여 셔플을 수행, 배치 크기로 데이터를 묶기
# 디코더의 입력과 결과시퀀스를 함께구성
# 디코더의 결과 시퀀스에서는 시작 토큰을 제거해야 한다.
def make_dataset(input, output, batch_size=64, buffer_size=20000):
    data_set = tf.data.Dataset.from_tensor_slices(({
            'inputs': input,
            'dec_inputs': output[:, :-1]},{  # 마지막 토큰 제거
            'outputs': output[:, 1:]},  # 시작토큰 제거, [CES], 결과 시퀀스에는 시작 토큰이 나올필요가 없음
    ))
    data_set = data_set.cache()
    data_set = data_set.shuffle(buffer_size)
    data_set = data_set.batch(batch_size)
    data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
    return data_set


# 토큰화 / 정수 인코딩 / 패딩
def tokenize_and_filter(inputs, outputs, MAX_LENGTH, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def preprocess_sentence(sentence):
    # 단어와 구두점 사이에 공백 추가.
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def evaluate(sentence, tokenizer, MAX_LENGTH, model):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, tokenizer, MAX_LENGTH, model):
    prediction = evaluate(sentence, tokenizer, MAX_LENGTH, model)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
MAX_LENGTH = 40

DATA_PATH = './data_in/'

data_chatbot = pd.read_csv(DATA_PATH + 'ChatbotData.csv', encoding='utf-8')
data_chatbot.dropna(axis=0, inplace=True)

question_chatbot = []
for sentence in data_chatbot['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    question_chatbot.append(sentence)

answer_chatbot = []
for sentence in data_chatbot['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answer_chatbot.append(sentence)


tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    question_chatbot + answer_chatbot, target_vocab_size=2 ** 13)

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# -> 이 경우 token의 양이 많고 시작, 종료와 같은 특수토큰이 이미 존재함
# 다만 한글의 경우 의미있게 잘린다고 판단을 내리기 힘듦, ex) 안녕하세요-> 안_, *녕, *하, *세, *요


START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dff=DFF,
                    d_model=D_MODEL, num_heads=NUM_HEADS,dropout=DROPOUT)


tf.keras.utils.plot_model(model, to_file='transformer_settings.png', show_shapes=True)

questions, answers = tokenize_and_filter(question_chatbot, answer_chatbot, MAX_LENGTH, tokenizer)

dataset = make_dataset(questions, answers)


learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


checkpoint_path_transformer = DATA_PATH + '/weights_transformer_chatbot.h5'
cp_callback_transformer = ModelCheckpoint(
    checkpoint_path_transformer, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
# model.load_weights('weights_transformer_chatbot.h5')

EPOCHS = 50
model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback_transformer])

output = predict(sentence="사랑에 대해 아니?", tokenizer=tokenizer, MAX_LENGTH=MAX_LENGTH, model=model)

model.save(DATA_PATH + "/chat_bot.h5")

# #############################################

data_abst = pd.read_csv(DATA_PATH + 'news_abstraction.csv', encoding='utf-8')

data_abst.dropna(axis=0, inplace=True)

print(data_abst.isnull().sum())

data_abst.head(10)

text = []
for sentence in data_abst['text']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    text.append(sentence)

abstraction = []
for sentence in data_abst['abstractive']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    abstraction.append(sentence)


tokenizer_abs = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(text + abstraction, target_vocab_size=2 ** 13)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# -> 이 경우 token의 양이 많고 시작, 종료와 같은 특수토큰이 이미 존재함
# 다만 한글의 경우 의미있게 잘린다고 판단을 내리기 힘듦, ex) 안녕하세요-> 안_, *녕, *하, *세, *요

START_TOKEN, END_TOKEN = [tokenizer_abs.vocab_size], [tokenizer_abs.vocab_size + 1]
VOCAB_SIZE = tokenizer_abs.vocab_size + 2


model_abst = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dff=DFF,
                         d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)

text, abstraction = tokenize_and_filter(text, abstraction, MAX_LENGTH, tokenizer_abs)

dataset_abst = make_dataset(text, abstraction)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model_abst.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

checkpoint_path_transformer = DATA_PATH + '/weights_transformer_news.h5'
cp_callback_transformer = ModelCheckpoint(
    checkpoint_path_transformer, monitor="loss", verbose=1, save_best_only=True, save_weights_only=True)

EPOCHS = 50
model_abst.fit(dataset_abst, epochs=EPOCHS, callbacks=cp_callback_transformer)

example_sentence = "과학기술정보통신부는 20일 한국 화학연구원에을 개최했다고 밝혔다.이번 협약은 코로나19 " \
                   "대응을 위해 한시적으로 운영하던 연구협력체계를 지속적으로 운영하고, 신변종 감염병 대응 " \
                   "주요 연구기관간 협력을 공고히 하기 위해 진행됐다. 협약에 따라 감염병 연구협력의 구심점 " \
                   "역할을 수행할 '바이러스 연구 협력협의체'를 발족하고, 감염병 치료제·백신 개발에 필수적인 " \
                   "전임상시험을 체계적으로 지원하기 위한 '국가 전임상시험 지원센터' 출범을 공표했다. " \
                   "바이러스 연구협력협의체는 지난 7월 개소한 한국바이러스기초연구소를 중심으로 바이러스 " \
                   "감염병을 연구하는 대학과 출연연이 참여, 연구역량을 결집하고 감염병 공동연구 기획·연구개발 " \
                   "전략 수립 등 기초·원천 R&D 수행에 있어 싱크탱크 역할을 해나갈 계획이다. 국가 전임상시험 지원센터는 " \
                   "기존 코로나19 대응 연구개발지원 협의체를 통해 긴급적이고 한시적으로 지원하던 치료제·백신의 유효성 검증과 " \
                   "독성 평가를 상시적이고 총괄적으로 지원하는 체계를 구축할 계획이다. 고서곤 연구개발정책실장은 \"바이러스 " \
                   "연구협력협의체와 국가 전임상시험 지원센터 출범을 계기로 대학과 기업을 아우른 다양한 연구주체 간의 협력이 " \
                   "활성화되어 미래 감염병에 대응할 수 있는 기초·원천연구 역량이 강화될 수 있을 것이라 기대한다\"고 말했다."
output = predict(sentence=example_sentence, tokenizer=tokenizer_abs, MAX_LENGTH=MAX_LENGTH, model=model_abst)
