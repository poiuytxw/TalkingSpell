import subprocess
import torch


# 定义 YOLO 训练命令
command = [
    "yolo", 
    "task=detect", 
    "mode=train", 
    # "model=yolo11n.pt",
    "pretrained=True",
    "model=runs/detect/train/weights/best.pt", 
    "data=D:/GithubDesktopClone/UIST/m_data/data.yaml", 
    "patience=25",
    "epochs=100", 
    "imgsz=320", 
    "device=0"
]

if __name__=='__main__':
    # 运行命令
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")