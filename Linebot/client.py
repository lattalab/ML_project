import requests

def read_file(file_path):   # 讀取檔案內容
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def send_text_to_server(text):
    # 這是你要上傳資料的網址
    url = 'your_website url'
    data = {'text': text}
    response = requests.post(url, json=data)
    print(response.json())

if __name__ == "__main__":
    # 更改你的檔案路徑
    file_path = 'your file path'
    # 得到檔案內容
    content = read_file(file_path)
    # 上傳到Linebot Webhook伺服器
    send_text_to_server(content)
