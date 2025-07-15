import cv2
import numpy as np
import requests
import os
# from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse


# url="http://192.168.137.49"

def show_points(coords,labels,ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# model = YOLO("YOLO/model/yolo11n.pt")
# cap = cv2.VideoCapture(0)

if __name__=='__main__':
    #添加参数
    parser = argparse.ArgumentParser(description='创建文件夹和子文件夹')
    parser.add_argument('folder_name', type=str, help='主文件夹的名称')
    parser.add_argument('url', type=str, help='ip地址')

    args = parser.parse_args()
    # set_resolution(url, index=3, verbose=True)
    frame_interval=2
    output_folder=os.path.join('../m_data', 'raw_img',args.folder_name)
    os.makedirs(output_folder, exist_ok=True)
    saved_count=0
    frame_count=0
    max_frames_to_save=100
    saving_enabled = False                   # 标记是否启用保存图像
    input_point = np.array([[60, 100]])
    input_label = np.array([1])

    cap = cv2.VideoCapture(args.url + ":81/stream")
    while True:
        if cap.isOpened():
            ret,frame=cap.read()
            if ret:
                cropped_frame = frame[0:320, 80:400]  # x范围从100到700，y范围从0到600
                # print(frame.shape)
                
                 # 添加圆形标记
                cropped_frame_with_mark=cropped_frame.copy()
                circle_center = (160, 160)  # 圆心位置
                cv2.circle(cropped_frame_with_mark, circle_center, 20, (255, 0, 0), -1)  # 半径为 20 的圆形
                cv2.imshow("240x240", cropped_frame_with_mark)


                # show_points(input_point, input_label, plt.gca())
                # 检测按键
                key = cv2.waitKey(1) & 0xFF  # 等待 1 毫秒并获取按键
                if key == ord('s'):  # 检测是否按下 's' 键
                    saving_enabled = not saving_enabled  # 切换保存状态
                if saving_enabled:
                    print("Started saving frames...")
                else:
                    print("Stopped saving frames.")

                #保存图片
                if saving_enabled and frame_count % frame_interval == 0:
                    # 确保图像尺寸为800x600
                    #这里应该要有一个trigger，按一个什么键然后开始进行记录
                    image_filename=os.path.join(output_folder,f'{saved_count:04d}.jpg')
                    cv2.imwrite(image_filename,cropped_frame)  # 保存图像
                    saved_count += 1  # 更新保存计数
                    if saved_count == max_frames_to_save:
                        print("Stopped saving after 50 frames.")
                        saving_enabled = False  # 自动停止保存
            frame_count+=1
            if not ret:
                break
    cv2.destroyAllWindows()
    cap.releae()


