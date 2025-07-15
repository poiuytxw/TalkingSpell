import cv2
import json
import os
from ultralytics import YOLO
from status import InfoStorage

# Load the YOLO11 model
model = YOLO("runs/detect/train8/weights/best.pt")
storage = InfoStorage()
# Open the video file
# video_path = "test.mp4"
# cap = cv2.VideoCapture(video_path)
url="http://192.168.137.12"
cap = cv2.VideoCapture(url + ":81/stream")
# cap = cv2.VideoCapture(0)


tts_clone=["0_old_woman","1_young_woman","2_kid_woman","3_old_man","4_young_man","5_kid_man","6_no_gender"]
old_detect="nothing"
new_detect="nothing"

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True)

        results=model(frame,conf=0.75)
        # results = model("https://ultralytics.com/images/bus.jpg")
        if results[0].boxes.cls.shape[0] == 0:
            print("detected nothing")
            storage.update_detection("nothing","unknown")
            new_detect="nothing"
        else:
            #检测到对应的内容
            idx=int(results[0].boxes.cls[0].item())
            detected=results[0].names[idx]
            new_detect=detected
            if os.path.exists(f'../../m_settings/char_{detected}.json'):
                char_path=f'../../m_settings/char_{detected}.json'
                with open(char_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    tts_id=data['rvc']
                    print("use tts:",tts_id,"conf:",results[0].boxes[0].conf)#好这里已经选定了tts了
                    tts_path="../../m_voice/clone_audio/"+tts_clone[tts_id]+".mp3"
            
        if new_detect!=old_detect:
            old_detect=new_detect
            storage.update_detection(detected,tts_id)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()