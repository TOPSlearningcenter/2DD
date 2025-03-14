from ultralytics import YOLO
import cv2

model_name = 'demo/yolov10b.pt'
model = YOLO(model_name)

video_path = 'demo/demo.mp4'
cap = cv2.VideoCapture(video_path)

window_name = 'Object Detection'
escape_key = 'q'

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    if len(results) == 0:
        cv2.imshow(window_name, frame)
    else:
        annotated_frame = results[0].plot()
        cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord(escape_key):
        break

cap.release()
cv2.destroyAllWindows()
