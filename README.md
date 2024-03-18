## Current Dataset:
ASVspoof 2019 dataset: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset  
Developing `PA` dataset only now. (2024/3/18)

## Code Preview
1. Transfer.py 用來將指定資料集的檔案按照指定格式轉成csv儲存
2. mel-frequency.ipynb 
* 分析CSV檔的資訊
* 得到資料夾下的資訊 (影片名稱、有多少個等等)
* An helpful function to change an audio to `mel-frequency diagram`.
* 真正做轉換 (有根據需求調整)
3. model.ipynb
* 先統計資料數量
* 利用 pytorch.imagefolder 包成 dataset (train, validation, test)
* Define CNN model
* Training and Testing

## Result
(2024/3/18) 模型在訓練階段的準確率可達到九成以上，缺點使用Eval當作測試資料時準確率會掉為八成以上  
改善方向: Test data 盡量貼近九成
