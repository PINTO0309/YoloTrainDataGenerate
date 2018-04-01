# [Japanese] YoloTrainDataGenerate
YoloV2独自学習データの生成＋Movidius Neural Compute Stick向け学習データコンバート

YoloV2 Generate original learning data + Learning data conversion for Movidius Neural Compute Stick

## 環境
(1)【学習用PC】 GIGABYE U2442F

  ・MEM：16GB

  ・CPU：第3世代 Intel Core i7-3517U(1.9GHz)

  ・GPU：Geforce GT 650M (VRAM:2GB)

  ・OS：Ubuntu 16.04 LTS (Windows10とのデュアルブート)

  ・CUDA 8.0.61

  ・cuDNN v6.0

  ・Caffe

  ・OpenCV 3.4.0

  ・Samba

(2)【実行環境】 Raspberry Pi 3 ModelB

  ・Raspbian Stretch

  ・NCSDK v1.12.00

  ・Intel Movidius Neural Compute Stick

  ・OpenCV 3.4.0

  ・Samba

## 流れ
  1. 適当に動画撮影
  2. 動画から機械的に静止画を大量生成
  3. 大量の静止画から物体部分を機械的に抽出して背景が透過した物体画像生成
  4. 別途用意した背景静止画と 3. で生成した物体静止画をランダムに回転・縮小・拡大・配置・ノイズ追加しながら合成して大量に水増し
  5. 学習
  6. Intel Movidius Neural Compute Stick 用学習データ(graph)へ変換
  7. Raspberry Pi上で 6. を使用してtinyYoloによる複数動体検知


