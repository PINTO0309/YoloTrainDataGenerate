# [Japanese] YoloTrainDataGenerate
YoloV2独自学習データの生成＋Movidius Neural Compute Stick向け学習データコンバート<br>
YoloV2 Generate original learning data + Learning data conversion for Movidius Neural Compute Stick<br>
https://qiita.com/PINTO/items/d5645734ca9c95b1c395

　
# 環境
* MEM：16GB
* CPU：第3世代 Intel Core i7-3517U(1.9GHz)
* GPU：Geforce GT 650M (VRAM:2GB)
* OS：Ubuntu 16.04 LTS
* CUDA 8.0.61
* cuDNN v6.0
* Caffe
* OpenCV 3.4.0
* Samba

　
# おおまかな流れ
1. 適当に動画撮影
2. 動画から機械的に静止画を大量生成
3. 大量の静止画から物体部分を機械的に抽出して背景が透過した物体画像生成
4. 別途用意した背景静止画と 3. で生成した物体静止画をランダムに回転・縮小・拡大・配置・ノイズ追加しながら合成して大量に水増し

　
# 動画→静止画変換

`$ ffmpeg -i xxxx.mp4 -vcodec png -r 10 image_%04d.png`

　
# 指定フォルダ内の複数静止画ファイル、複数物体周囲をまとめて機械的に透過加工

* 背景が白色に近い色・物体が白色／灰色以外の配色で構成されている場合のみ動作
* １画像内に複数物体が写っている場合は物体数分の画像ファイルへ分割して加工
* 入力画像が長方形であっても最終生成画像は物体を含む96×96の正方形
* エッジ抽出の都合上、重なり合っている物体は１つと認識される
* 検出された物体の面積が1000pxに満たない場合は当該物体を抽出対象から除外
* 最終生成された画像内に物体が存在しないと判断される場合はファイルを生成しない
```
$ cd YoloTrainDataGenerate
$ python3 object_extraction.py
```
(1) 編集元画像 1920x1080<br>
&nbsp;&nbsp;&nbsp;&nbsp;![1.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/1.png)<br>
(2) 元画像の背景白色化 1920x1080<br>
&nbsp;&nbsp;&nbsp;&nbsp;![2.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/2.png)<br>
(3) 物体検出 1920x1080<br>
&nbsp;&nbsp;&nbsp;&nbsp;![3.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/3.png)<br>
(4) 背景透過処理後PNGファイル２枚 96x96<br>
&nbsp;&nbsp;&nbsp;&nbsp;![4.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/4.png)&nbsp;&nbsp;&nbsp;&nbsp;![5.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/5.png)![6.png](https://github.com/PINTO0309/YoloTrainDataGenerate/blob/master/media/6.png)
<br>

# 画像の前処理
images_org配下のファイル名をpyrenamer等を利用して「(ラベル名)_xxxx.png」に一括変更
* xxxx の箇所は同一ラベル名で重複しないように連番なり、文字列なり、自由に設定(４桁でなくても良い)

```（例）.
　labelA_0001.png　→　「labelA」に集約
　labelA_0002.png　→　「labelA」に集約
　labelA_0003.png　→　「labelA」に集約
　labelB_0001.png　→　「labelB」に集約
　labelB_0002.png　→　「labelB」に集約
　labelC_0001.png　→　「labelC」に集約
　　　：
```

```（例）.ラベルがswitchとremoconの場合
　switch_0001.png　→　「switch」に集約
　switch_0002.png　→　「switch」に集約
　switch_0003.png　→　「switch」に集約
　switch_0004.png　→　「switch」に集約
　remocon_0001.png　→　「remocon」に集約
　remocon_0002.png　→　「remocon」に集約
　　　：
```
<br>

# 学習用画像データ他の自動生成

* 静止画からランダムに回転・縮小・拡大・配置・ノイズ追加を繰り返して水増し画像生成
* 任意の物体画像と任意の背景画像を自由に合成
* 前処理で生成した連番付き画像ファイル名の連番部を無視し、複数画像をひとつのラベルへ集約
* train.txt、test.txt、label.txt が生成される
* images配下に水増し済みの静止画像が生成される
* デフォルトの水増し枚数は10,000枚、変更する場合は generate_sample.py の 「train_images = 10000」 を修正する


下記コマンドを実行
```
$ python3 generate_sample.py
```
