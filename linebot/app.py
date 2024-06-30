from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
from flask import Flask, request, abort, jsonify
import re

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
        line_bot_api.push_message(user_id, message)

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

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=10000)