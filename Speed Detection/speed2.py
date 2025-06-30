import cv2
from ultralytics import YOLO
from tracker import Tracker
import time

# Parameter
MODEL_PATH       = "yolov8n.pt"
VIDEO_PATH       = "veh2.mp4"
CLASS_WHITELIST  = {'car', 'truck', 'bus'}
DISTANCE_M       = 10           
SPEED_LIMIT_KMH  = 120
OFFSET_PX        = 6          
MID_X_SPLIT      = 510          

# Garis atas bawah
cy1_l, cy2_l = 322, 368          # kiri
cy1_r, cy2_r = 322, 368          # kanan

model   = YOLO(MODEL_PATH)
cap     = cv2.VideoCapture(VIDEO_PATH)
fps     = cap.get(cv2.CAP_PROP_FPS)
tracker = Tracker()

vh_down, vh_up      = {}, {}
done_down, done_up  = set(), set()
violators           = set()
prev_cy             = {}         

# Callback mouse
def show_xy(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"x:{x}  y:{y}")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", show_xy)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))  
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() 

    # box kendaraan
    bboxes = []
    for (x1, y1, x2, y2, conf, cls) in detections:
        if model.names[int(cls)] in CLASS_WHITELIST:
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    tracked = tracker.update(bboxes)

    # Proses setiap kendaraan
    for x1, y1, x2, y2, vid in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # set garis
        if cx < MID_X_SPLIT:        # LANE KIRI
            cy1, cy2 = cy1_l, cy2_l
        else:                       # LANE KANAN
            cy1, cy2 = cy1_r, cy2_r

        # Simpan & hitung arah
        direction = 0
        if vid in prev_cy:
            direction = cy - prev_cy[vid]  
        prev_cy[vid] = cy

        # Down
        if direction > 0:
            # Crossing garis pertama
            if cy1 - OFFSET_PX < cy < cy1 + OFFSET_PX:
                vh_down[vid] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)  

            # Crossing garis kedua → hitung speed
            if vid in vh_down and cy2 - OFFSET_PX < cy < cy2 + OFFSET_PX:
                t0 = vh_down.pop(vid)
                t1 = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_s = (t1 - t0) / fps
                if vid not in done_down and elapsed_s > 0:
                    done_down.add(vid)
                    speed = (DISTANCE_M / elapsed_s) * 3.6  # m/s → km/h
                    color = (0, 0, 255) if speed >= SPEED_LIMIT_KMH else (0, 255, 0)
                    if speed >= SPEED_LIMIT_KMH:
                        violators.add(vid)

                    cv2.putText(frame, f'{int(speed)} Km/h', (x2, y2),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        # Up
        elif direction < 0:
            if cy2 - OFFSET_PX < cy < cy2 + OFFSET_PX:
                vh_up[vid] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1) 

            if vid in vh_up and cy1 - OFFSET_PX < cy < cy1 + OFFSET_PX:
                t0 = vh_up.pop(vid)
                t1 = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_s = (t1 - t0) / fps
                if vid not in done_up and elapsed_s > 0:
                    done_up.add(vid)
                    speed = (DISTANCE_M / elapsed_s) * 3.6
                    color = (0, 0, 255) if speed >= SPEED_LIMIT_KMH else (0, 255, 0)
                    if speed >= SPEED_LIMIT_KMH:
                        violators.add(vid)

                    cv2.putText(frame, f'{int(speed)} Km/h', (x2, y2),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        # Gambar box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, str(vid), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

    # Gambar garis referensi
    # Lane kiri
    cv2.line(frame, (0, cy1_l), (MID_X_SPLIT, cy1_l), (255, 255, 255), 1)
    cv2.line(frame, (0, cy2_l), (MID_X_SPLIT, cy2_l), (255, 255, 255), 1)
    cv2.putText(frame, 'L1-L', (10, cy1_l - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    cv2.putText(frame, 'L2-L', (10, cy2_l - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)

    # Lane kanan
    cv2.line(frame, (MID_X_SPLIT, cy1_r), (1020, cy1_r), (255, 255, 255), 1)
    cv2.line(frame, (MID_X_SPLIT, cy2_r), (1020, cy2_r), (255, 255, 255), 1)
    cv2.putText(frame, 'L1-R', (MID_X_SPLIT + 10, cy1_r - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    cv2.putText(frame, 'L2-R', (MID_X_SPLIT + 10, cy2_r - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)

    # Overlay 
    cv2.putText(frame, f'down: {len(done_down)}', (60, 90),
                cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 255), 2)
    cv2.putText(frame, f'up: {len(done_up)}', (60, 130),
                cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 255), 2)
    cv2.putText(frame, f'violators: {len(violators)}', (60, 170),
                cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # Esc untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
