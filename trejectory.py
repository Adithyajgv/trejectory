import cv2 as cv
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kalman = cv.KalmanFilter(6, 2)
        dt = 1
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 1], #x-pos
            [0, 1, 0, dt, 0, 0.5*dt**2], #y-pos
            [0, 0, 1, 0, dt, 0], #x-vel
            [0, 0, 0, 1, 0, dt], #y-vel
            [0, 0, 0, 0, 1, 0], #x-acc
            [0, 0, 0, 0, 0, 1] #y-acc
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
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)

    def Estimate(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kalman.correct(measured)
        predicted = self.kalman.predict()
        return (int(predicted[0]), int(predicted[1]))


cap = cv.VideoCapture('test.mp4')
filter = KalmanFilter()

lower_color = np.array([100, 150, 50])  
upper_color = np.array([140, 255, 255])

kernel = np.ones((5, 5), np.uint8)

total_error = 0
valid_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_frame, lower_color, upper_color)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv.contourArea(contour) > 500:
            x1, y1, w, h = cv.boundingRect(contour)
            x2 = x1 + w
            y2 = y1 + h
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            fx, fy = filter.Estimate(center_x, center_y)
            
            # prediction:
            cv.rectangle(frame, (fx - w // 2, fy - h // 2), (fx + w // 2, fy + h // 2), (0, 0, 255), 2)



            error = np.sqrt((fx - center_x) ** 2 + (fy - center_y) ** 2)
            total_error += error
            valid_frames += 1

    cv.imshow("Ball Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# Calculate the average error and print it at the end
if valid_frames > 0:
    average_error = total_error / valid_frames
    print(f"Average prediction error: {average_error:.2f} pixels")
else:
    print("No valid frames with detected objects.")
