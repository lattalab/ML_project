from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.models import AudioMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent, MessageAction
import os
from flask import Flask, request, abort, jsonify, send_from_directory
import subprocess
# from P_model_7 import CNN_model7
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from observe_audio_function import load_audio, get_mel_spectrogram, plot_mel_spectrogram, envelope, normalize_audio, SAMPLE_RATE, AUDIO_LEN
# from observe_audio_function import denoise, process_audio
# from PIL import Image

app = Flask(__name__)

# 輸入你的Line Bot的Token和Secret
# add channel access token
channel_access_token =  os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
line_bot_api = LineBotApi(channel_access_token)
# add channel secret
channel_secret =  os.environ.get('LINE_CHANNEL_SECRET')
# add user id
user_id =  os.environ.get('LINE_USER_ID')
handler = WebhookHandler(channel_secret)

@app.route('/')
def hello_world():
    return "hello world"

# 新增的 API 路徑，用來接收來自 client.py 送出的資料
@app.route('/receive_text', methods=['POST'])
def receive_text():
    data = request.get_json()   # 解析資料
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    # 處理接收到的文字資料
    send_message(text)
    # 向客戶端回應HTTP response，200代表成功
    return jsonify({"message": "Text received successfully"}), 200

# 送出指定訊息
def send_message(text):
    messages = []
    # 先增加起始訊息
    messages.append(
        TextSendMessage(text="開始啟用警告訊息傳輸功能....\n收到Vitis-ai推論後的結果了。"))
    # 以下內容需根據你的輸出檔自行決定
    L = text.split(';')     # 取出後面的分類值
    print(L)

    fake, c1 = L[0].split(':')
    real, c2 = L[1].split(':')
    if (c2 >= c1):
        print("該段音訊被判斷為真實語音")
        messages.append(TextSendMessage(text="該段音訊被判斷為真實語音"))
    else:
        print("該段音訊被判斷為合成語音")
        messages.append(TextSendMessage(text="該段音訊被判斷為合成語音"))

    for message in messages:
        # 傳送訊息到指定用戶(add your user_id)

        # 傳送訊息到指定用戶
        # broadcast 推播給所有好友
        line_bot_api.broadcast(message)

@app.route("/callback", methods=['POST'])
def callback():
    # 獲取X-Line-Signature header值
    signature = request.headers['X-Line-Signature']

    # 獲取請求正文
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print("Request body: " + body)  # 添加這行以便在控制台查看請求內容

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# 暫存使用者音檔路徑的字典
user_audio_path = {}

# 處理語音訊息事件
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    # 得到音檔內容
    audio_message_content = line_bot_api.get_message_content(event.message.id)
    # 紀錄音檔內容
    user_id = event.source.user_id
    audio_path = f'audio_{user_id}.mp3'

    # 保存音檔到伺服器
    with open(audio_path, 'wb') as fd:
        for chunk in audio_message_content.iter_content():
            fd.write(chunk)

    # 保存音檔路徑到暫存字典
    user_audio_path[user_id] = audio_path
    
    # 回覆選擇語言的按鈕
    buttons_template = ButtonsTemplate(
        title='選擇語言',
        text='請選擇你要辨識的語言',
        actions=[
            PostbackAction(label='中文', data='language=chinese'),
            PostbackAction(label='英文', data='language=english'),
            MessageAction(label='其他', text='抱歉! 目前不支援其他語言，請嘗試中英文語音。', data='language=other'),
        ]
    )
    # 送出這個樣板訊息
    template_message = TemplateSendMessage(alt_text='選擇語言', template=buttons_template)
    # 回傳給後端伺服器知道
    line_bot_api.reply_message(event.reply_token, template_message)

# 處理語言選擇後的事件
@handler.add(PostbackEvent)
def handle_postback(event):
    # 知道使用者選到的語言
    if event.postback.data.startswith('language='):
        selected_language = event.postback.data.split('=')[1]
        user_id = event.source.user_id

        if user_id in user_audio_path and selected_language in ['chinese', 'english']:
            audio_path = user_audio_path[user_id]
            #result_text = process_audio(audio_path, selected_language)
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='123')
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='未找到對應的音檔，請重新上傳。')
            )

@app.route('/download_audio/<user_id>', methods=['GET'])
def download_audio(user_id):
    if user_id in user_audio_path:
        return send_from_directory(directory='.', filename=user_audio_path[user_id], as_attachment=True)
    else:
        return "音檔不存在。", 404

# # 語音處理函數
# def process_audio(audio_path, language):
#     # 使用相應的模型判斷是否為合成語音
#     is_synthetic = check_synthetic_audio(audio_path, language)
#     return f"語言: {language}, 懷疑 {'是' if is_synthetic else '不是'} 合成語音"

# # 判斷是否為合成語音的函數
# def check_synthetic_audio(audio_path, language):
#     # 根據語言選擇相應的模型
#     if language == 'chinese':
#         pass
#         #model = load_model('chinese_synthetic_model')
#     elif language == 'english':
#         model = CNN_model7()    # 建立模型物件
#         # 加載權重檔案
#         weights_path = 'model_en.pth'
#         model.load_state_dict(torch.load(weights_path))
#         # 測試GPU可不可用
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Test on {device}.")
#         # 將模型設置為評估模式
#         model.to(device)
#         model.eval()
    
#     # 音檔要先轉換成頻譜圖
#     spectrogram(audio_path)
#     return model

# def spectrogram(audio_path):
#     spec_paths = []     # 用來保存頻譜圖的路徑
#     audio, sr = load_audio(audio_path, sr=SAMPLE_RATE)
#     for i in range(0, len(audio), AUDIO_LEN):
#         audio = audio[i:i+AUDIO_LEN]
#         rn = denoise(audio, sr=SAMPLE_RATE) # 新增去雜音
#         spec = get_mel_spectrogram(rn)
#         fig = plot_mel_spectrogram(spec)
#         plt.title("Spectrogram", fontsize=17)
#         # Save the spectrogram image with a meaningful filename
#         filename = f"spec_{i/AUDIO_LEN}.png"  # Use single quotes inside the f-string
#         filepath = os.path.join('./', filename)
#         plt.savefig(filepath)
#         spec_paths.append(filepath)

#         # Close the figure to free up resources
#         plt.close()
#         return spec_paths

# # Function to predict using the model
# def predict(model, image_path):
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # Resize images to expected size
#         transforms.ToTensor()          # Convert images to PyTorch tensors
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     # Move image to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image = image.to(device)

#     # Perform prediction
#     with torch.no_grad():
#         output = model(image)

#     return output

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=10000)
