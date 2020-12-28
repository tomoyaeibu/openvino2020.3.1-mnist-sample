#######################################################################################
#%% Initialize.
#
#

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import numpy as np
import os
from pathlib import Path

import time

#######################################################################################
#%% Utility.
#
#

def backup_raw(imarray, filepath): # float64
    backup = imarray.tobytes()

    with open(filepath, "wb") as fout:  
        fout.write(backup)

    return backup

def convert_kerasmodel_to_frozen_pb(kerasmodelpath, pbmodelname):
    output_pb = os.path.splitext(os.path.basename(pbmodelname))[0] + ".pb"
    output_pb_path = Path(output_pb)

    #%% Reset session
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    model = tf.keras.models.load_model(kerasmodelpath, compile=False)
    session = tf.compat.v1.keras.backend.get_session()

    input_names = sorted([layer.op.name for layer in model.inputs])
    output_names = sorted([layer.op.name for layer in model.outputs])

    graph = session.graph

    #%% Freeze Graph
    with graph.as_default():
        # Convert variables to constants
        graph_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graph.as_graph_def(), output_names)
        # Remove training nodes
        graph_frozen = tf.compat.v1.graph_util.remove_training_nodes(graph_frozen)

        with open(output_pb, 'wb') as output_file :
            output_file.write(graph_frozen.SerializeToString())

        print ('Inputs = [%s], Outputs = [%s]' % (input_names, output_names))

#######################################################################################
#%% Load data.
#
#

# mnistデータをダウンロードする。
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# [0,1]に収まるよう正規化する。
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape, x_test.shape); print()

#######################################################################################
#%% Setting model.
#
#

# クラス分類のモデルを定義する。
model = Sequential([
    Conv2D(50, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(50, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# モデルをコンパイルする。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#######################################################################################
#%% Training.
#
#

# 学習時にベストなモデルだけを保存するように設定する。
modelCheckpoint = ModelCheckpoint(filepath = 'model.h5',
                                  monitor = 'val_loss',
                                  verbose = 1,
                                  save_best_only = True,)
# Early Stoppingを設定する。
EarlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

# 学習を実行する。
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=5, verbose=1,
                 callbacks=[modelCheckpoint, EarlyStopping])

#######################################################################################
#%% Evaluation
#
#

# ベストなモデルをロードする。
best_model = load_model('model.h5')

# OpenVINOの結果と比較できるように入力データを保存し、推論結果を出力する。
backup_raw(x_test[5], 'x_test[5].raw')
np.set_printoptions(suppress=True)
print(x_test[5].shape)

start = time.perf_counter()
score_result = best_model.predict(x_test)[5]
end = time.perf_counter()
print("Time taken for inference : [{0}] ms".format(end-start))
print(score_result) 

# OpenVINOのオプティマイザで変換できるようにfrozon_pb形式でモデルを保存する。
convert_kerasmodel_to_frozen_pb("model.h5", "model.pb")

