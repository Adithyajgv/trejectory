import cv2 as cv
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import threading

class KalmanFilter:
    def __init__(self):
        self.kalman = cv.KalmanFilter(6, 2)
        dt = 1
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 1],  # x-pos
            [0, 1, 0, dt, 0, 0.5*dt**2],  # y-pos
            [0, 0, 1, 0, dt, 0],          # x-vel
            [0, 0, 0, 1, 0, dt],          # y-vel
            [0, 0, 0, 0, 1, 0],           # x-acc
            [0, 0, 0, 0, 0, 1]            # y-acc
        ], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)

        self.kalman.processNoiseCov = np.array([
            [1e-3, 0, 0, 0, 0, 0],
            [0, 1e-3, 0, 0, 0, 0],
            [0, 0, 1e-2, 0, 0, 0],
            [0, 0, 0, 1e-2, 0, 0],
            [0, 0, 0, 0, 1e-1, 0],
            [0, 0, 0, 0, 0, 1e-1]
        ], np.float32)

        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        #self.kalman.errorCovPost = np.eye(6, dtype=np.float32)

        self.initializing = True

    def Estimate(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        
        self.kalman.correct(measured)

        if self.initializing:
            self.kalman.statePost = np.array([[coordX], [coordY], [0], [0], [0], [0]], np.float32)
            self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 5
            self.initializing = False

        predicted = self.kalman.predict()
        return (int(predicted[0]), int(predicted[1]))




# process of iding and labling each object: function for multithreding
def process(box, class_id, score, track_id):
    global total_error
    global valid_frames
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if track_id not in KalmanDict:
        with lock:
            KalmanDict[track_id] = KalmanFilter()

    filter = KalmanDict[track_id]

    fx, fy = filter.Estimate(center_x, center_y)


    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"class: {class_id}, score: {score:.2f}, ID: {track_id}"
    cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.rectangle(frame, (fx - w // 2, fy - h // 2), (fx + w // 2, fy + h // 2), (0, 0, 255), 2)

    error = np.sqrt((fx - center_x) ** 2 + (fy - center_y) ** 2)

    with lock:
        total_error += error
        valid_frames += 1
        track = track_history[track_id]
        track.append((center_x, center_y))
        if len(track) > 15:
            track.pop(0)


#error vars:
total_error = 0
valid_frames = 0


#YOLO model:
model = YOLO("yolo11s.pt")
cap = cv.VideoCapture('multiple_test.mp4')

#ID tracking:
track_history = defaultdict(lambda: [])
track_ids = list()
lock = threading.Lock()
KalmanDict = dict()

#"main"
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="botsort.yaml")

    for result in results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls
        scores = result.boxes.conf
        try:
            track_ids = result.boxes.id.float().tolist()
        except:
            continue
        
        thread_list = list()
        for box, class_id, score, track_id in zip(boxes, class_ids, scores, track_ids):
            thread = threading.Thread(target=process, args=(box, class_id, score, track_id))
            thread_list.append(thread)
            thread.start()
        cv.imshow("Object Detection and Tracking", frame)

        for thread in thread_list:
            thread.join()


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

#err calculation 
if valid_frames > 0:
    average_error = total_error / valid_frames
    print(f"Average prediction error: {average_error:.2f} pixels")
else:
    print("No valid frames with detected objects.")
