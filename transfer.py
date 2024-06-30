import csv

# 用來生成csv檔案的函數

def txt_to_csv(input_file, output_file):

    # 定義要加入的列名稱
    header = ["SPEAKER_ID", "AUDIO_FILE_NAME", "ENVIRONMENT_ID", "ATTACK_ID", "KEY(Classification)"]

    with open(input_file, 'r') as txtfile, open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 寫入標題列
        csv_writer.writerow(header)

        for line in txtfile:
            # 假設每行的資料以空格分隔，您可以根據實際情況調整分隔符號
            row = line.strip().split(' ')
            csv_writer.writerow(row)

# 請將以下路徑替換為您的txt檔案和要寫入的csv檔案路徑
txt_file_path = './dataset/PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019_PA_cm_dev_trl.txt'
csv_file_path = './dataset/PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019_PA_cm_dev_trl.csv'

txt_to_csv(txt_file_path, csv_file_path)
print("TXT to CSV finish")