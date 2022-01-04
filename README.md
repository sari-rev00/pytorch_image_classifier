## イントロダクション
画像データを分類するCNNモデルの作成と評価を行います。
CNNフレームワークとしてpytorchを使用します。<br>
GPUを使用せず、CPUのみで動作します。
<br>
<br>

## 動作概要
- ラベル付けされた画像データを学習用(train)・検証用(test)に分割する
- 上記を教師データとして学習する
- 以下の学習スコアをグラフとして表示・保存する
    - loss
    - 予測精度(acc)
- 学習済みモデルの重みデータの保存・読み込みを行う
- 画像データからラベルを予測（推論）する

使用方法はclassify_mnist_data.ipynbを参照してください。<br>
<br>

## データセットの形式
ディレクトリに画像ファイル（.jpeg、.jpg）を入れたものをデータセットとして使用します。<br>
各ディレクトリ内の画像のクラスは同じになるようにします。<br>
<br>

## 各モジュールの機能
- DataFile (utils/dataloader.py)
    - 入力：　画像ファイルが入ったのフォルダのパスと、そのフォルダ内の画像のラベルのリスト
    - 出力：　個々の画像のパスとラベルのリスト
         - split=Trueでデータを学習(train)用、検証(test)用に分割して出力します。
         - 検証用データはtest_sizeで指定された比率で元のデータからランダムに選定されます。
- Dataset (utils/dataloader.py)
    - 入力：　画像データのパスとラベルのリスト
    - 出力：　画像データ（画素・色チャンネルのマトリックス）、画像のラベル
- Dataloader (utils/dataloader.py)
    - 学習時に画像データ・ラベルの組をバッチサイズぶん供給する
- gen_dataloader (utils/dataloader.py)
    - DataFile、Dataset、Dataloader間のデータ受け渡しを内包したメソッド
    - 入力：　画像ファイルが入ったのフォルダのパスと、そのフォルダ内の画像のラベルのリスト
    - 出力：　Dataloaderインスタンス、データのdescription
<br><br>

- MNIST (models/cnn.py)
    - MNIST画像データ（手書きアラビア数字）の推論モデル
    - 学習済み推論モデルの重みデータの保存
    - 学習済み推論モデルの重みデータの読み込み
<br><br>

- Manager (utils/model_manager.py)
    - 推論モデルの学習(train())
        - num_epoch：　エポック数
        - print_epoch_step：　精度・lossの表示を何epoch毎に実施するか
    - 学習スコアデータの保存・グラフ化
    - 学習済み推論モデルの重みデータの保存　＊モデルclassの機能の呼び出し
        - train(auto_save=True)にすると、学習時に自動で保存する。
    - 学習済み推論モデルの重みデータの読み込み　＊モデルclassの機能の呼び出し
    - 推論の実行
<br><br>

## Configファイルの設定内容 (config/config.py)
**ConfDataloader**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| BATCH_SIZE | 学習バッチのサイズ（データ数） | 8 |
| SHUFFLE | エポック毎にデータの順序を入れ替える | True |
| DROP_LAST | エポックの最後のバッチのデータ数がBATCH_SIZEに満たない場合、そのバッチは学習に使用しない | True |
| TARGET_EXT | 対応可能な学習データファイルの拡張子 | [".jpg", ".jpeg"] |
| RANDOM_SEED | SHUFFLEで使用する乱数のシード | 12345 |
<br>

**ConfManager**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| ACC_TH | 精度(acc)がこの値を超えた場合にモデルパラメタの自動保存を行う | 0.98 |
| SAVE_DIR_BASE | モデルパラメタを保存するディレクトリ | "./weight_data/" |
<br>

**ConfOptimizer(torch.optim.SGD)**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| LEARNING_RATE | 学習率 | 1e-4 |
| MOMENTUM | momentum factor | 0.9 |
| WEIGHT_DECAY | 正則化(L2)パラメタ | 1e-5 |
<br>

**TransformParam**<br>
画像データと使用するモデルに応じて変更します。
下記はMNIST画像データとMNIST用CNNモデルを想定した値です。
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| resize | 変換後の画像のサイズ（ピクセル） | 28 |
| color_mean |  | [0.5] |
| color_std |  | [0.5] |
<br>

## System requirements (Dev environment)
- **OS:** Win10 (or latter)
- **Hardware** Intel CORE i5(8th gen)
- **Python:** 3.7 (or latter)
<br><br>

## Setting
Clone this repositoty into your system, and move to root directory of repository.<br>
And install requirements.
```
$(win) pip install -r requirements.txt
```
Download MNIST data as image files.<br>
\* open "download_mnist_data_as_files.ipynb" and run all(2) cells.<br>
You can specify total data amount with "data_num", and result shows data amount of each label (0, 1, 2 ...) 
```
from utils.download_mnist_data import download_mnist_image_files
from pprint import pprint

d_count = download_mnist_image_files(data_num=10000)
pprint(d_count)

>>>
preparing img files...
removed stored img
Completed
{'0': 1001,
 '1': 1127,
 '2': 991,
 '3': 1032,
 '4': 980,
 '5': 863,
 '6': 1014,
 '7': 1070,
 '8': 944,
 '9': 978}
```
<br>

## Usage
Open "classify_mnist_data.ipynb", it can help you to understand usage of utilities.<br>
### cell 1
Import required libraries.
### cell 2
Prepare list of directories (contain MNIST image data files) and labels.<br>
Adjust number of labels (calsses) of MNIST data and make daraloader instance.<br> 
Show number of train and test data.<br>
### cell 3
Extruct data (of index 1) from dataloader, and desplay it.<br>
### cell 4 and 5
Prepare model instance.<br>
### cell 6
Prepare manager instance and train model.<br>
### cell 7
Show training result.<br>
### cell 8
Save trained model weight data.<br>
### cell 9
Make another manager instance from model weight data above.<br>
### cell 10
Make prediction form 10 data from test dataset and compare with labels of data.<br>
<br>

## Misc
Copyright (c) 2022 SAri<br>



