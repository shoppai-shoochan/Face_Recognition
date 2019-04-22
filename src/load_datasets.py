import pickle
import cv2
import numpy as np

#pickleに保存したデータセットロード
def LoadDataset(path):
    with open(path,mode='rb') as f:
        image_data = pickle.load(f)         #numpyの画像データをロード
        label_data = pickle.load(f)         #numpyのラベル(数値)データをロード
        index_to_label = pickle.load(f)     #ラベル(数値）→ラベル(文字列)にするdict型データをロード

    data_len = len(image_data)              #データ数を取得
    split_rate = 0.9
    train_len = int(data_len * split_rate)  #訓練画像はデータ数の9割とする
    np.random.seed(30)                      #検証のため、乱数の設定を30で固定する
    indice = np.random.permutation(range(data_len))     #データを訓練用とテスト用に分離
    x_train = image_data[indice[0:train_len]]
    y_train = label_data[indice[0:train_len]]
    x_test = image_data[indice[train_len:data_len]]
    y_test = label_data[indice[train_len:data_len]]
    labels = set(label_data)
    class_num = len(labels)     #クラス数を取得

    print(data_len)
    print(x_train.shape)
    print(x_test.shape)
    print(index_to_label)
    print(str(class_num) + 'クラス')

    return (x_train,y_train),(x_test,y_test),class_num,index_to_label

#以下はデバック用のコード
if __name__ == '__main__':
    path = 'ここにpickleファイルのパスを入力する'
    (x_train,y_train),(x_test,y_test),class_num,index_to_label = LoadDataset(path)
    for i in range(100):
        print(index_to_label[y_test[i]])    #ラベルを出力
        cv2.imshow("test",x_test[i])        #画像を表示
        cv2.waitKey(0)                      #windowにキー入力があるまで待機



