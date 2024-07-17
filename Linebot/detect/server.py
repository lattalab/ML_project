# 一些linebot有關跟flask的library
from linebot import LineBotApi, WebhookHandler, LineBotSdkDeprecatedIn30
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage
import os   # os library , easy to find filepath
from flask import Flask, request, abort, jsonify
from linebot.models import VideoMessage, AudioMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent, MessageAction
from dotenv import load_dotenv  # environment variable
import warnings     # 忽略警告
import subprocess   # run other python script
import pandas as pd # read csv
from pydub import AudioSegment  # fetch sound from video

# 不知道先載完library有沒有比較快 (感覺有)
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
import numpy as np
import time
import torch
import torch.nn as nn
import sys

app = Flask(__name__)
load_dotenv()   # 引入環境變數

# 輸入你的Line Bot的Token和Secret
# add channel access token
channel_access_token =  os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
line_bot_api = LineBotApi(channel_access_token)
# add channel secret
channel_secret =  os.getenv('LINE_CHANNEL_SECRET')
# add user id
user_id =  os.getenv('LINE_USER_ID')
handler = WebhookHandler(channel_secret)

# 測試主畫面(會印出hello world)
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
# source folder
source_folder = "./audio/"
os.makedirs(source_folder, exist_ok=True)   # 創建資料夾

# 處理語音訊息事件
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    # 得到音檔內容
    audio_message_content = line_bot_api.get_message_content(event.message.id)
    # 紀錄音檔內容
    user_id = event.source.user_id  # 傳送訊息的該使用者ID
    audio_name = f'audio_{user_id}.wav'
    audio_path = os.path.join(source_folder, audio_name)

    # 保存音檔
    with open(audio_path, 'wb') as fd:
        for chunk in audio_message_content.iter_content():
            fd.write(chunk)
    fd.close()

    print(f"Saving audio to {audio_path}")
    print("Audio saved successfully")
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

# 處理影片訊息事件
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    # 得到影片內容
    video_message_content = line_bot_api.get_message_content(event.message.id)
    # 紀錄影片內容
    user_id = event.source.user_id  # 傳送訊息的該使用者ID
    video_name = f'audio_{user_id}.wav'
    video_path = os.path.join(source_folder, video_name)

    # 保存影片
    with open(video_path, 'wb') as fd:
        for chunk in video_message_content.iter_content():
            fd.write(chunk)
    fd.close()

    print(f"Saving video to {video_path}")
    print("Video saved successfully")

    # 將影片轉換成音檔
    song = AudioSegment.from_file(video_path)
    song.export(video_path, format="wav")   # 重寫影片成純音檔
    
    print("convert video to \"audio\" successfully!")

    # 保存音檔路徑到暫存字典
    user_audio_path[user_id] = video_path
    
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
            # 先送出一些訊息說要等
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text='判斷過程，需要一點時間，請稍後...')
            )
            # 執行判斷
            result_text = process_audio(audio_path, selected_language, os.path.basename(audio_path))
            # 送出結果
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=result_text)
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='未找到對應的音檔，請重新上傳。')
            )

# 語音處理函數
def process_audio(audio_path, language, filepath):
    # 先轉成Spectrogram
    subprocess.run(['python', 'create_spectrogram.py', filepath])
    # 使用相應的模型判斷是否為合成語音
    subprocess.run(['python', 'test_model.py', language])
    # 讀取結果 (1: spoof , 0: bonafide)
    is_synthetic = check_synthetic_audio(audio_path, language)
    return f"語言為:{language}\n 該段音訊可能 \"{'是' if is_synthetic else '不是'}\" 合成語音"

# 得出模型判斷結果的函數
def check_synthetic_audio(audio_path, language):
    csv_file_path = f"output_model.csv"
    df = pd.read_csv(csv_file_path)
    return df['classified'].value_counts().idxmax() # 取出最多的分類值 (0 or 1)
    
if __name__ == "__main__":
    # 忽略叫你用linebot.v3的警告。  
    warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)
    app.run(host='127.0.0.1', port=10000)