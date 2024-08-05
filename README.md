# How to read?
接下來會逐一解釋每個目錄在做甚麼，通常以列點式說明每個項目大概在什麼。
- [ ] 最終版本

## Base directory
Dataset: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

currently developed on `PA` dataset.

**code description (start at 2024/3/18)**
1. transfer.py: 為資料集狀態描述生成一個csv
2. mel-frequency.ipynb: 先分析資料集狀態，來決定要怎麼取資料，在轉換為頻譜圖
3. model.ipynb: 開始訓練英文模型
4. 其餘檔案為模型權重檔以及頻譜圖範例。

### Result
希望準確率能達到九成，目前在eval資料集下的準確率表現是最好84%。

## Linebot (start at 2024/6/15)
1. 去Line Developement 建立一個LineBot
2. 回應設定:Webhook要打開 (用來設定callback網址用的)
3. 找到屬於LineBot的CHANNEL_ACCESS_TOKEN 跟 CHANNEL_SECRET 及 USER_ID
4. 可以在Line Developement做簡單的reply功能

i. 準備好.py檔 (跑你想要的做的事情)  
ii. 上傳到Github  
iii. 使用Render做為雲端用來佈署  

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

render.yaml 包含部署到 Render 需要的資訊，像是範例程式如何 build、如何啟動。  (套別人的github模板)  
requirements.txt 程式所需要的套件集  
Build Command:./build.sh  (會執行build.sh的功能)  

**設計理念**
把這個當作專題當中vitis-ai的後處理。 (目前算完成)
由client.cpp or client.py 上傳資料到指定網站上，在啟動Linebot警告功能。(但不是最終vitis-ai的執行檔)

### Result
能成功上板執行，得出警示作用。
在client執行成功會得到:
![image](https://github.com/lattalab/ML_project/assets/91266449/a2c274e3-6496-4676-8a7d-ea70869eda96)

![image](https://github.com/lattalab/ML_project/assets/91266449/5690c754-157d-48be-8c44-7fea073e252e)

### update (2024/07/18)
原本的Linbot資料夾會被部屬到render上，但之後想要新增線上判斷合成語音的功能，部屬到render上遇到瓶頸。  
於是新增了`detect`資料夾，來重新開發。  
* server.py: 在原本app.py的架構下進行額外開發，能夠對音檔(或影片音訊)判斷是否為合成語音
* create_spectrogram.py : 將音檔轉換為Spectrogram (前處理的過程)
* test_model.py : 轉換完spectrogram之後，送到模型上進行預測，預測完後會寫成一個csv檔(方便自己debug用、檢查數據用)
* 其他資料: 模型架構、模型權重、測試音檔等相關檔案

```
使用ngrok來產生網址 --> ngrok http <your-server-IP>:<your-server-Port> in Anaconda
run server.py at your compiler(ex: VSCode)
Send message to your linebot
```

#### Result-2
![image](https://github.com/user-attachments/assets/26b10796-7596-4bab-94bf-821818d9b925)
![image](https://github.com/user-attachments/assets/c0303f44-0ded-40c8-876d-d627351480ce)

### update-3 - 0806
先前說到能讓Linebot判斷合成語音，更新了:  
* 優化前處理過程
* 優化效能，採用了一些平行處理的方式，讓文本分析跟判斷合成語音過程可以同時做
* 優化介面，會詢問使用者意見，根據使用者所選的選項進行動作  
![image](https://github.com/user-attachments/assets/91c5519f-6ec6-47af-a670-eb377d71162c)  
* 使用`Gemini API`結合文本分析，使用者能選擇是否要判斷合成語音中的文本內容  
![image](https://github.com/user-attachments/assets/5b868fd6-b1ec-4f47-b0b8-dde92c2d6072)  

## CH_AiSound (2024/6/24)
* spectrum.py & model.ipynb: 開始訓練分辨中文合成語音模型
資料來自`CFAD`跟自行生成的語音(elevanlabs、TTSMAKER)
* spectrum.cpp: 嘗試用C++寫出python librosa轉換成頻譜圖功能，引用其他公開程式碼  
  ref: https://github.com/ewan-xu/LibrosaCpp
* fake & real : 模型訓練時用到的頻譜圖

### Result
vitis-ai可以跑python，~~暫定不需要研究C++轉成頻譜圖。~~   
跑太慢了，還是要研究看看。  

### Update (2024/7/12)
因為轉圖片的前處理太久(10s~20s)，因此被認為還是需要研究出怎麼用C/C++寫出頻譜圖，在`CH_AiSound`底下新增`news`資料夾代表新增測試程式
* spectrogram.cpp : 利用先前的librosaCpp + gnuplot嘗試畫出spectrogram
* spectrogram_with_fftw.cpp : 利用<fftw3>現成library實現STFT + gnuplot畫出spectrogram
* sample.py : 原始轉換頻譜圖檔案
* 有圖片比照
* 結果: 雖然CPP非常難以近似python librosa的結果但是有達到非常類似的效果了。  

### Update2 (2024/7/20)
在`CH_AiSound/news/update2`底下加入新增檔案，主要內容為優化先前的C語言Spectrogram寫法，並寫了一些檔案做數據比對，最終結果可以直接比較圖，有達到非常近似的效果了。  
缺點: 相比原始python轉圖片檔案，沒有對音訊長度做剪裁、過濾掉強度較小的部分。

## denoise (start at 2024/6/30)
Make use of `noisereduce` library to achieve denoise feature.

Further:
* remove echo sound
* 離麥克風距離遠近可能影響收音 (目前假設都盡量在近的地方收音)

### Result
暫定用noisereduce去雜音效果不錯，其他問題會影響模型再來調適。  

## analysis (2024/8/6)
* process_audio.ipynb: 對三種目前常用的訓練特徵(MFCC、Chroma、Mel-spectrogram)進行分析畫分布圖  
* other_method.ipynb : 考慮其他可能用到的聲音特徵，依樣進行分析畫分布圖  
* t_test.ipynb : 用t檢定比較 合成語音跟真實語音在某特定特徵下分布差異 (平均是否相同)  
* scatter.ipynb: 將高維特徵進行降維畫散佈圖，看是否能找出差異  
