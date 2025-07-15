# UIST-2025 Talking Spell  
## 一个可以通过相机和任何物品对话的项目  
## 目录
`pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`
- [video_streaming.ino] (基本上和esp32 webserver项目相同)
- [app_httpd.cpp] (webUI)
## Hardware
Seeed studio esp32s3 sense
![image](https://github.com/user-attachments/assets/5aa33ee6-9f8a-45db-aee6-8cc7e970bfa6)
## Reference
### Hardware
- [seeed studio wiki] https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/
- [camera usage] https://wiki.seeedstudio.com/xiao_esp32s3_camera_usage/#project-ii-video-streaming
### AI Voice
- &#x2705; [RVC] https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- [GPT-SoVITS] https://github.com/RVC-Boss/GPT-SoVITS
- [SVC] https://github.com/svc-develop-team/so-vits-svc
- [Tutorial] https://www.youtube.com/watch?v=59A5uxIKw1s

### RVC
- [人声模型下载] https://huggingface.co/QuickWick/Music-AI-Voices/tree/main
- [音乐下载] https://learnerprofiler.co.za
- [Beta 版本] https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main

- [UVR5] https://github.com/Anjok07/ultimatevocalremovergui

### Object Detection
- [Target Detection] https://github.com/ejcgt/attention-target-detection

### Segment Anything
- [Transformer] https://arxiv.org/abs/1706.03762
- [SAM] https://github.com/facebookresearch/segment-anything
- &#x2705;[SAM2] https://github.com/facebookresearch/sam2
- [FastSAM] https://github.com/CASIA-IVA-Lab/FastSAM
- [EfficientSAM] https://github.com/yformer/EfficientSAM
- [EfficientTAM] https://github.com/yformer/EfficientTAM

### YOLO(You only look once)
- &#x2705; [YOLO] https://arxiv.org/abs/1506.02640v5s
  - https://www.youtube.com/watch?v=svn9-xV7wjk
- [YOLO-World] https://github.com/ailab-cvc/yolo-world

### CLIP
- [QWEN-VL] 慢

### STT
- [Fast-whisper] https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file
  - [BUG] Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path! 
  `pip install "ctranslate2<4.5.0"`
### TTS
- [edge-tts]

### Debug
- [esp32 cam webserver本地streaming] https://www.hackster.io/onedeadmatch/esp32-cam-python-stream-opencv-example-1cc205



### Log
[2025-2-21] &#x2705; opencv获取在本地获取视频流 // RVC测试 // CLIP本地单图测试

