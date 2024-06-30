## Base directory
Dataset: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
currently developed on `PA` dataset.

### code description (start at 2024/3/18)
1. transfer.py: 為資料集狀態描述生成一個csv
2. mel-frequency.ipynb: 先分析資料集狀態，來決定要怎麼取資料，在轉換為頻譜圖
3. model.ipynb: 開始訓練英文模型

其餘檔案為模型權重檔以及頻譜圖範例。

### Result
希望準確率能達到九成，目前在eval資料集下的準確率表現是最好84%。

## Linebot (start at 2024/6/15)
1. 去Line Developement 建立一個LineBot
2. 回應設定:Webhook要打開
3. 找到屬於LineBot的CHANNEL_ACCESS_TOKEN 跟 CHANNEL_SECRET
4. 可以在Line Developement做簡單的reply功能

i. 準備好.py檔 (用來執行複雜的/rate 跟 匯率換算)  
ii. 上傳到Github  
iii. 使用Render做為雲端用來佈署  

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

render.yaml 包含部署到 Render 需要的資訊，像是範例程式如何 build、如何啟動。  (套別人的github模板)
requirements.txt 程式所需要的套件集  
Build Command:./build.sh  (會執行build.sh的功能)  

### 設計理念
由client.cpp上傳資料到指定網站上，在啟動Linebot警告功能。(但不是最終vitis-ai的執行檔)

### Result
能成功上板執行，得出警示作用。

![image](https://github.com/lattalab/ML_project/assets/91266449/5690c754-157d-48be-8c44-7fea073e252e)

## CH_AiSound (2024/6/24)
* spectrum.py & model.ipynb: 開始訓練分辨中文合成語音模型
資料來自`CFAD`跟自行生成的語音(elevanlabs、TTSMAKER)
* spectrum.cpp: 嘗試用C++寫出python librosa轉換成頻譜圖功能，引用其他公開程式碼
ref: https://github.com/ewan-xu/LibrosaCpp
* fake & real : 模型訓練時用到的頻譜圖

### Result
vitis-ai可以跑python，暫定不需要研究C++轉成頻譜圖。

## denoise
TODO
