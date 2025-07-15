from ultralytics  import YOLOWorld
# from ultralytics import YOLO
import cv2
import supervision as sv

from supervision.assets import download_assets, VideoAssets

# Download a supervision video asset
# path_to_video = download_assets(VideoAssets.PEOPLE_WALKING)
# frame_generator = sv.get_video_frames_generator(path_to_video)
# frame = next(frame_generator)
# cv2.imshow("Image",frame)

model = YOLOWorld("YOLO/model/yolov8m-worldv2.pt")
# model.set_classes(["toy"])

bounding_box_annotator=sv.BoundingBoxAnnotator()
lable_annotator=sv.LabelAnnotator()

# cap=cv2.VideoCapture('people-walking.mp4')
cap = cv2.VideoCapture(0)
w,h,fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FPS))

out=cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))

while cap.isOpened():
    ret,img=cap.read()
    if not ret:
        break
    results=model.predict(img)
    detections=sv.Detections.from_ultralytics(results[0])
    annotated_frame=bounding_box_annotator.annotate(
        scene=img.copy(),
        detections=detections
    )
    annotated_frame=lable_annotator.annotate(
        scene=annotated_frame,detections=detections
    )
    out.write(annotated_frame)
    cv2.imshow("Image",annotated_frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()