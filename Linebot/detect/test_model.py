from P_model_13_ENG import CNN_model13, crop_with_points, transform
from P_model_9_CH import CNN_model9
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

dict1 = {
	"Image_name":[],
	"output_0":[],
	"output_1":[],
    "classified":[]
}

def data_for_model(image_folder_path):
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]

    # Apply Transformations
    # get the cropped image
    # Convert the NumPy array back to an image
    images = [transform(crop_with_points(path).convert('RGB')) for path in image_paths]
    # images = [transform(Image.open(path).convert('RGB')) for path in image_paths]

    # 隨機指派一個標籤，這個不重要，要看得是模型的預測輸出
    label = np.array([0 for _ in range(len(images))])   

    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(label, dtype=torch.long))

    # Create DataLoader for train and validation sets
    # batch size = 1 , 因為一張一張跑
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1 , shuffle=False, pin_memory=True)

    return data_loader

def test_model(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        for _, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device) , label.to(device)
            output = model(image)

            # record the outputs of each columns
            dict1["output_0"].extend(output[:, 0].tolist())
            dict1["output_1"].extend(output[:, 1].tolist())
            dict1["classified"].extend(output.argmax(dim=1).tolist())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <language>")
        sys.exit(1)

    # access language
    language = sys.argv[1]

    # test device whether gpu can use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test on {device}.")

    model = None
    # load model depend on language
    if language == "english":
        model = CNN_model13()    # define model
        state_dict = torch.load('./model_13_ENG_ver1.pth')
    else:
        model = CNN_model9()    # define model
        state_dict = torch.load('./model_9_CH_ver1.pth')
    
    model.to(device)
    model.load_state_dict(state_dict)

    # add marks to print the layer data info
    # register_hooks(model) 

    # image_folder_path = r"C:\Users\User\Desktop\code\dataset_testing"
    image_folder_path = "./spec/"

	# record the Image_name
    dict1["Image_name"] = [filename for filename in os.listdir(image_folder_path)]
	
	
    test_dataloader = data_for_model(image_folder_path)
    print(f"Loaded test data from {image_folder_path}.")

    print("Testing...")
    test_model(model, test_dataloader, device)
    
    # turn dict1 into csv file
    output_df = pd.DataFrame(dict1)
    csv_file_path = f"output_model.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")
