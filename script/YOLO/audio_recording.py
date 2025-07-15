import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from status import InfoStorage
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

##faster whisper
from faster_whisper import WhisperModel
model_size = "large-v3"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")


from openai import AzureOpenAI
client=AzureOpenAI(
    api_key="7bf86d32229a45ed8d486fc5aa2a3370",
    api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net"
)

# 替换为您的 ESP32 的 IP 地址
esp32_ip = "http://192.168.137.12"  # 例如 http://192.168.1.100

duration = 5  # 最大录音时长（秒）
sample_rate = 44100  # 采样率
audio_data = None  # 用于存储录音数据
IsRecording=False
storage = InfoStorage()


import json
def update_json(msg,file_name):
    # 指定要写入的 JSON 文件名
    # 将 messages 写入 JSON 文件
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(msg, json_file, ensure_ascii=False, indent=4)

    print(f"消息已写入 {file_name}")

import torch
from TTS.api import TTS
# import threading
# import queue
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


# def generate_audio(text,tts_path,language,audio_queue):
#     """生成音频并放入队列"""
#     data = tts.tts(text=text, speaker_wav=tts_path, language=language)
#     audio_np = np.array(data, dtype=np.float32)
#     audio_queue.put(audio_np)  # 将生成的音频放入队列

# def play_audio(audio_queue):
#     """从队列中播放音频"""
#     while True:
#         audio_np = audio_queue.get()  # 获取队列中的音频
#         if audio_np is None:  # 检查是否结束
#             break
#         sd.play(audio_np, samplerate=22050)
#         sd.wait()  # 等待音频播放完成

import re
def split_text_by_marker(long_text, marker="[QAQ]"):
    """根据标识符切分长文本"""
    # 使用正则表达式切分文本
    parts = re.split(re.escape(marker), long_text)
    return [part.strip() for part in parts if part.strip()]  # 去掉空白部分



m_messages=[]

# 录音函数
def record_audio(json_path,tts_path):
    print("recording...")
    global audio_data
    # button.description = '正在录音...'
    # button.disabled = True
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 等待录音完成
    if audio_data is not None:
        # 保存为 WAV 格式
        print("writing...")
        sf.write('../../m_voice/user/recording.mp3', audio_data, sample_rate)
        whisper(json_path,tts_path)

    else:
        print("请先录音！")

def whisper(json_path,tts_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        m_messages = json.load(file)
    segments, info = model.transcribe("../../m_voice/user/recording.mp3", beam_size=5,language="zh")
    msg=""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        msg=msg+" "+segment.text
    print(msg)
    m_messages.append({"role":"user","content":msg})
    response=client.chat.completions.create(
    model="gpt-35-turbo",
    messages=m_messages,)   
    print(response.choices[0].message.content)
    # tts.tts(text=response.choices[0].message.content, speaker_wav=tts_path, language="zh")
   
    # audio_queue = queue.Queue()
    # playback_thread = threading.Thread(target=play_audio(audio_queue))
    # playback_thread.start()
    chunks=split_text_by_marker(response.choices[0].message.content)
    for chunk in chunks:
        print(chunk)
        wav=tts.tts(chunk, speaker_wav=tts_path, language="zh")
        sd.play(wav,samplerate=22050)
        sd.wait()
    #     generate_audio(chunk,tts_path,'zh',audio_queue)
    # audio_queue.put(None)  # 发送结束信号
    # playback_thread.join()  # 等待播放线程完成

    #play the sound
    m_messages.append({"role":"assistant","content":response.choices[0].message.content})
    if len(m_messages)==13:
        #进行了5轮对话之后
        m_messages.pop(2)
        m_messages.pop(1)
    update_json(m_messages,json_path)



def GetBlueToothStatus(file_path='recording_state.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # result=data['state']
    return data['state']

import keyboard    
def get_ldr_value():
    time.sleep(1)
    # try:
    #     response = requests.get(f"{esp32_ip}/ldr")  # 向 /ldr 路由发送请求
    #     if response.status_code == 200:
    #         ldr_value = response.text
    test=GetBlueToothStatus()
    # print(f"光敏电阻读数: {test}")  # 输出光敏电阻值
    if test=="start recording...": 
        print(storage.get_detection())
        IsRecording=True#可以开始复印
        if(storage.get_detection()!="nothing"):
            MessageJSON=storage.get_history()
            CharJSON=storage.get_char()
            TTSMP3=storage.get_tts()
            print("I got:",MessageJSON,CharJSON,TTSMP3)
            #初始化聊天记录
            m_messages=[]
            #如果没有MessageJson，说明第一次见
            if os.path.exists(MessageJSON)==False:
                with open(CharJSON, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                m_messages.append({"role":"system","content":"请根据下列json中的内容确认自己的人设："+str(data)+"请扮演这个角色并参考背景故事和用户对话，但在对话中不要直接复制json中的内容，请以朋友的身份和用户对话，你的回复不要超多50字,如果回复的内容中有涉及到背景故事中没有的部分你可以自由发挥，请在回复内容中的每一个换气或是停顿的位置，使用[QAQ]标识符将其隔开，每个气口之间最好不要超过10个字，请尽量用多个短句来回复"})
                update_json(m_messages,MessageJSON)
            else:
                with open(CharJSON, 'r', encoding='utf-8') as file:
                    m_messages = json.load(file)

            # signal_response = requests.post(f"{esp32_ip}/signal", data="recording")
            # if signal_response.status_code == 200:
            #     print("信号已发送给 ESP32")
            # else:
            #     print("发送信号失败")
            record_audio(MessageJSON,TTSMP3)  
    #     else:
    #         print("无法获取数据，状态码:", response.status_code)
    # except Exception as e:
    #     print("请求失败:", e)

if __name__ == "__main__":
    # whisper()
    while True:
        get_ldr_value()