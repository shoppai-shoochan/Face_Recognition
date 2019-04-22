import keras
import numpy as np
from load_datasets import LoadDataset
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal, Constant

#データセットの読み込み
path = 'ここにpickleファイルのパス'
(x_train,y_train_n),(x_test,y_test_n),class_num,index_to_label = LoadDataset(path)  #データセット読み込み
train_len = x_train.shape[0]    #訓練データ数
test_len = x_test.shape[0]      #テストデータ数
x_train = x_train.reshape(train_len,224,224,1).astype(int)  #行列の形を(データ数,224,224)→(データ数,224,224,1)
x_test = x_test.reshape(test_len,224,224,1).astype(int)     #要素の型をuint→int
#x_train,x_test = x_train/255.0,x_test/255.0
y_test = np.eye(class_num)[y_test_n.astype(int)]        #ラベル（整数値）をone-hotのベクトル
y_train = np.eye(class_num)[y_train_n.astype(int)]


#畳み込み層定義用
def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

#全結合層定義用
def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

#ディープラーニングのモデルを作成
def AlexNet(class_num):     #class_num分類のalexnetを作成する
    ROWS = 224      #入力ピクセル数(横軸）
    COLS = 224      #入力ピクセル数(縦軸）
    CHANNELS = 1    #チャンネル数は１（グレースケール画像）
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(class_num, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#ディープラーニングの学習及び評価
model = AlexNet(class_num)
model.compile(loss="categorical_crossentropy",optimizer=SGD(),metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=128,epochs=10,validation_split=0.2,verbose=1)  #ディープラーニングの学習
score = model.evaluate(x_test,y_test,verbose=0)     #テストデータの評価
print("Test loss:",score[0])
print("Test accuracy:",score[1])

#作成したモデル及び学習した重みを保存
model_json = model.to_json()    #モデルをjson形式文字列に
path = "./alexnet_model.json"   #モデルを保存するファイルパス
with open(path, mode='w') as f:
    f.write(model_json)
model.save_weights('./alexnet_arashi_mlp_weights.h5')   #学習した重みを保存