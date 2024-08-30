# 整理梅爾頻譜跟模型測試的function
# 並設計成可以直接呼叫，這樣只要傳這個檔案就可以了
import create_spectrogram
import test_model
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def Save_mel(filename):
    os.makedirs(create_spectrogram.destination_folder, exist_ok=True)  # 存放spectrogram的資料夾
    
    print("Remove old images...")
    for i in os.listdir(create_spectrogram.destination_folder):    # 刪除舊的圖片
        i_path = os.path.join(create_spectrogram.destination_folder, i)
        os.remove(i_path)
    
    print("image filename:", filename)
    filepath = os.path.join(create_spectrogram.source_folder, filename)    # 從這邊讀取音檔
    create_spectrogram.process_audio_file(filepath)

    print(f"Spectrogram images saved to {create_spectrogram.destination_folder}")

def Get_Predict(language):
    # test device whether gpu can use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test on {device}.")

    model = None
    # load model depend on language
    if language == "english":
        model = test_model.CNN_model13()    # define model
        state_dict = torch.load('./model_13_ENG_ver1.pth')
    else:
        model = test_model.CNN_model9()    # define model
        state_dict = torch.load('./model_9_CH_ver1.pth')
    
    model.to(device)
    model.load_state_dict(state_dict)

    # add marks to print the layer data info
    # register_hooks(model) 

    # image_folder_path = r"C:\Users\User\Desktop\code\dataset_testing"
    image_folder_path = "./spec/"

	# record the Image_name
    # 定義變數dict1
    dict1 = {
	"Image_name":[],
	"output_0":[],
	"output_1":[],
    "classified":[]
    }
    dict1["Image_name"] = [filename for filename in os.listdir(image_folder_path)]
	
	# 轉變成模型可以讀取的資料
    test_dataloader = test_model.data_for_model(image_folder_path)
    print(f"Loaded test data from {image_folder_path}.")

    print("Testing...") # 執行模型測試
    model.eval()
    with torch.no_grad():
        for _, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device) , label.to(device)
            output = model(image)

            # record the outputs of each columns
            dict1["output_0"].extend(output[:, 0].tolist())
            dict1["output_1"].extend(output[:, 1].tolist())
            dict1["classified"].extend(output.argmax(dim=1).tolist())
    
    # turn dict1 into csv file
    output_df = pd.DataFrame(dict1)
    csv_file_path = f"output_model.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")