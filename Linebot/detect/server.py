# 一些linebot有關跟flask的library
from linebot import LineBotApi, WebhookHandler, LineBotSdkDeprecatedIn30
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage,TextMessage
from linebot.models import VideoMessage, AudioMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent
import os   # os library , easy to find filepath
from flask import Flask, request, abort, jsonify    # json
from dotenv import load_dotenv  # environment variable
import warnings     # 忽略警告
import subprocess   # run other python script
import pandas as pd # read csv
import google.generativeai as genai # use gemini
import speech_recognition as sr # speech recognition
from pydub import AudioSegment  # audio processing
from pydub.utils import make_chunks  # split audio
import pathlib  # path (Read audio file and send to Gemini)

# downloading youtube audio
from pytubefix import YouTube
from pytubefix.cli import on_progress
import threading    # threading for speed up
from linebot.v3.messaging import (  # loading animation
    ShowLoadingAnimationRequest,
    Configuration,
    AsyncApiClient,
    AsyncMessagingApi
    ) 
import asyncio  # async function

# 預先載入library (因為感覺這樣比較快)
from wrapFuncForDL import Save_mel, Get_Predict

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
# ADD gemini api key
gemini = os.getenv('GEMINI_API_KEY')

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
split_folder = "./split_audio/"
os.makedirs(source_folder, exist_ok=True)   # 創建資料夾
os.makedirs(split_folder, exist_ok=True)   # 創建資料夾

# 處理語音訊息事件
@handler.add(MessageEvent, message=(AudioMessage, VideoMessage))
def handle_audio_message(event):
    # 得到音檔內容
    audio_message_content = line_bot_api.get_message_content(event.message.id)
    # 紀錄音檔內容
    user_id = event.source.user_id  # 傳送訊息的該使用者ID
    audio_name = f'audio_{user_id}.wav'
    audio_path = os.path.join(source_folder, audio_name)

    for i in os.listdir(source_folder):    # 刪除舊的音檔
        i_path = os.path.join(source_folder, i)
        os.remove(i_path)

    # 保存音檔
    with open(audio_path, 'wb') as fd:
        for chunk in audio_message_content.iter_content():
            fd.write(chunk)
    fd.close()

    print(f"Saving audio to {audio_path}")
    print("Audio saved successfully")
    # 保存音檔路徑到暫存字典
    user_audio_path[user_id] = audio_path
    send_language_selection(event)
    
# 處理語言選擇事件
def send_language_selection(event):
    buttons_template = ButtonsTemplate(
        title='選擇語言',
        text='請選擇你要辨識的語言\n注意!! 判斷合成語音的過程需要一點時間且你選擇的語言會影響結果',
        actions=[   # 選項
            PostbackAction(label='中文', data='language=chinese'),
            PostbackAction(label='英文', data='language=english'),
            PostbackAction(label='其他', data='language=other'),
        ]
    )
    template_message = TemplateSendMessage(
        alt_text='選擇語言',
        template=buttons_template
    )
    line_bot_api.reply_message(event.reply_token, template_message)

# 處理執行文本分析事件
def send_analysis_selection(user_id, selected_language):
    # 根據選擇的語言顯示第二個選單
    buttons_template = ButtonsTemplate(
        title='是否執行文本分析?',
        text='請選擇是否執行對語音內容進行潛在詐騙內容分析',
        actions=[   # 選項
            PostbackAction(
                label='是 (執行)',
                data=f'action=yes&language={selected_language}'
            ),
            PostbackAction(
                label='否 (不執行)',
                data=f'action=no&language={selected_language}'
            )
        ]
    )
    template_message = TemplateSendMessage(
        alt_text='選擇分析選項',
        template=buttons_template
    )
    line_bot_api.push_message(user_id, template_message)

@handler.add(PostbackEvent)
def handle_postback(event):
    data = event.postback.data
    user_id = event.source.user_id  # 獲取用戶ID
    audio_path = user_audio_path[user_id]   # 得到儲存檔案的路徑

    if data.startswith('language='):
        selected_language = data.split('=')[1]  # 獲取選擇的語言
        filename = os.path.split(audio_path)[1]  # 得到檔案名稱


        # 同時詢問是否要繼續執行文本分析
        send_analysis_selection(user_id, selected_language)

        # 開始處理音頻
        # process_thread = threading.Thread(target=process_audio, args=(user_id, selected_language, filename))
        # process_thread.setDaemon(True)
        # process_thread.start()
        process_audio(user_id, selected_language, filename)
        print("Start to process audio to detect AI Generated Sound!!!")

    elif data.startswith('action='):    # 獲取是否執行文本分析的選項
        asyncio.run(waiting_animation(user_id=user_id))    # 顯示等待動畫
        # 切割data
        action = data.split('&')[0].split('=')[1]
        language = data.split('&')[1].split('=')[1]
        if action == 'yes': # 需要執行文本分析
            # 開啟新線程來進行文本分析
            # analysis_thread = threading.Thread(target=scam_analyze, args=(user_id, audio_path, language))
            # analysis_thread.setDaemon(True)
            # analysis_thread.start()
            scam_analyze(user_id, audio_path, language)
            print("Start to analyze the text content for scam!!!")
        elif action == 'no':    # 不執行文本分析
            print("不執行文本分析")
            pass

# 語音處理函數
def process_audio(user_id, language, filename):
    # 先轉成Spectrogram
    # subprocess.run(['python', 'create_spectrogram.py', filename])
    Save_mel(filename)

    # 使用相應的模型判斷是否為合成語音
    # subprocess.run(['python', 'test_model.py', language])
    Get_Predict(language)
    
    # 讀取結果 (1: spoof , 0: bonafide) (來自csv檔的結果)
    print("讀取結果中:")
    csv_file_path = f"output_model.csv" # 因為被儲存在當前路徑下
    df = pd.read_csv(csv_file_path) # read csv file
    is_synthetic = df['classified'].value_counts().idxmax() # 取出最多的分類值 (0 or 1)
    # 設定訊息回復方式
    Language = '中文' if language == 'chinese' else '英文'  # 將Postback所得到的資料language轉換成繁體中文
    msg = f"語言為:{Language}\n該段音訊可能 \"{'是' if is_synthetic else '不是'}\" 合成語音"
    # 回傳訊息到Linebot上
    line_bot_api.push_message(
                user_id, TextSendMessage(text=msg)
            )

# 呼叫Gemini來幫助判斷文本內容是否為詐騙訊息
def scam_analyze(user_id, audio_path, selected_language, speech=True, msg=""):
    path = audio_path
    lang = {'chinese': 'zh-TW', 'english': 'en-US'} # 語言參數給speech recognition
    text = ""  # 要送入模型判斷的文本

    # 因為發現有更好用的方法，因此不再使用這個方法
    # if speech:  # 是語音檔
    #     #轉檔
    #     AudioSegment.converter = './ffmpeg/ffmpeg/bin/ffmpeg.exe'
    #     # 切割音檔
    #     chunk_files = split_audio(path)

    #     #辨識
    #     r = sr.Recognizer()
    #     count =0
    #     for path_i in chunk_files:  # 逐一辨識
    #         with sr.AudioFile(path_i) as source:
    #             audio = r.record(source)
    #         try:
    #             text += r.recognize_google(audio, language=lang[selected_language])
    #         except:    # 有時候語音太模糊會無法辨識
    #             count += 1
    #             pass

    #     if count >= len(chunk_files)/2: # 無法辨識超過太多，視為錯誤
    #         text = "發生了錯誤，無法辨識"
    #         line_bot_api.push_message(
    #                 user_id,
    #                 TextSendMessage(text=text)
    #             )
    #         return
    # else:   # 是文字檔
    #     text = msg

    # Access your API key as an environment variable.
    genai.configure(api_key=gemini)
    # Choose a model that's appropriate for your use case.
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """I need assistance in analyzing text for scam detection. 
    The response should include a structured analysis report which was divided into 6 parts: 
    giving a brief summary for given input text, indicating the likelihood of a scam 
    ,denoting which type of scam it is and analyzing the reason by point 
    ,giving some preventional advise and last give a simple conclusion.
    Please ensure that the response follows the specified structure for ease of parsing and integration with the application.
    You need to reply in Traditional Chinese in default.
    input text: """ + text
    if speech:  # 語音檔
        # Generate content based on the prompt.
        response = model.generate_content([
            prompt,
            {
            "mime_type": "audio/wav",
            "data": pathlib.Path(audio_path).read_bytes()
            }
        ])
    else:   # 文字檔
        response = model.generate_content([prompt + msg])

    line_bot_api.push_message(
                user_id,
                TextSendMessage(text=response.text)
            )

# 處理網址語音事件 (eg: youtube)
@handler.add(MessageEvent, message=TextMessage)
def handle_url(event):
    # 得到訊息內容
    message_content = event.message.text
    if not(message_content.startswith("https")):    # 如果不是網址，則處理為文字
        # 文字檔的語言不重要(因為不用speech recognition)
        scam_analyze(event.source.user_id, message_content, "chinese", speech=False, msg=message_content)
        return
    
    user_id = event.source.user_id  # 傳送訊息的該使用者ID
    audio_name = f'audio_{user_id}.wav'
    audio_path = os.path.join(source_folder, audio_name)

    # on_progress 看進度條
    yt = YouTube(message_content, on_progress_callback = on_progress)
    
    ys = yt.streams.get_audio_only()    # fetch audio only
    ys.download(output_path=source_folder, filename=audio_name)
    # 保存音檔路徑到暫存字典
    user_audio_path[user_id] = audio_path
    send_language_selection(event)

# 因為語音辨識有長度限制，因此用切分音檔的方式來處理
def split_audio(audio_path, chunk_length_ms=20000):
    """將音訊檔案分割成小片段"""
    print("開始切分音檔")
    audio = AudioSegment.from_file_using_temporary_files(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)  # 使用 make_chunks 來分割音訊
    chunk_files = []
    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.wav"
        chunk_name = os.path.join(split_folder, filename)
        chunk.export(chunk_name, format="wav")
        chunk_files.append(chunk_name)
        print(f"Exported {chunk_name}")
    return chunk_files

# 點擊按鈕後的回覆動畫
async def waiting_animation(user_id):
    configuration = Configuration(
        access_token=channel_access_token
    )

    async_api_client = AsyncApiClient(configuration)
    line_bot_api = AsyncMessagingApi(async_api_client)
    # loading second default is 5
    await line_bot_api.show_loading_animation(ShowLoadingAnimationRequest(chatId=user_id, loadingSeconds=5))

if __name__ == "__main__":
    # 忽略叫你用linebot.v3的警告。  
    warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)
    app.run(host='127.0.0.1', port=10000)