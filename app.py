import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math
import requests

API_URL = "http://127.0.0.1:8001/update"

SEND_EVERY_N_FRAMES = 5
frame_count = 0
# ============================================
# CONFIG
# ============================================

def send_to_server(data):

    try:
        requests.post(API_URL, json=data, timeout=0.1)
    except:
        pass  # don't break real-time loop

CAMERA_INDEX = 0

# Eye
EAR_THRESHOLD = 0.23

# Mouth / Yawning
MAR_THRESHOLD = 0.60
YAWN_FRAMES = 15

# Head Turn (Yaw)
HEAD_YAW_THRESHOLD = 0.18

# Head Tilt / Nod
HEAD_TILT_THRESHOLD = 12

# Risk thresholds
PRE_DROWSY_RISK = 3
DROWSY_RISK = 6

# Smoothing
SMOOTH_WINDOW = 15
BLINK_WINDOW_SEC = 60

# ============================================
# MEDIAPIPE
# ============================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# LANDMARKS
# ============================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 291, 13, 14]

NOSE = 1

LEFT_FACE = 234
RIGHT_FACE = 454

FOREHEAD = 10
CHIN = 152

# ============================================
# HELPERS
# ============================================

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def pt(lm, i, w, h):
    return int(lm[i].x * w), int(lm[i].y * h)

# ============================================
# EAR
# ============================================

def ear(lm, eye, w, h):

    p1 = pt(lm, eye[0], w, h)
    p2 = pt(lm, eye[1], w, h)
    p3 = pt(lm, eye[2], w, h)
    p4 = pt(lm, eye[3], w, h)
    p5 = pt(lm, eye[4], w, h)
    p6 = pt(lm, eye[5], w, h)

    vertical = dist(p2, p6) + dist(p3, p5)
    horizontal = 2 * dist(p1, p4)

    return vertical / horizontal

# ============================================
# MAR
# ============================================

def mar(lm, w, h):

    top = pt(lm, MOUTH[2], w, h)
    bottom = pt(lm, MOUTH[3], w, h)

    left = pt(lm, MOUTH[0], w, h)
    right = pt(lm, MOUTH[1], w, h)

    return dist(top, bottom) / dist(left, right)

# ============================================
# HEAD YAW
# ============================================

def head_yaw(lm, w, h):

    nose = pt(lm, NOSE, w, h)

    left = pt(lm, LEFT_FACE, w, h)
    right = pt(lm, RIGHT_FACE, w, h)

    center_x = (left[0] + right[0]) / 2

    face_width = dist(left, right)

    return (nose[0] - center_x) / face_width

# ============================================
# HEAD TILT / NOD
# ============================================

def head_tilt(lm, w, h):

    forehead = pt(lm, FOREHEAD, w, h)
    chin = pt(lm, CHIN, w, h)

    dx = chin[0] - forehead[0]
    dy = chin[1] - forehead[1]

    angle = math.degrees(math.atan2(dx, dy))

    return angle

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ============================================
# STATE VARIABLES
# ============================================

ear_hist = deque(maxlen=SMOOTH_WINDOW)
mar_hist = deque(maxlen=SMOOTH_WINDOW)
yaw_hist = deque(maxlen=SMOOTH_WINDOW)
tilt_hist = deque(maxlen=SMOOTH_WINDOW)

blink_times = deque(maxlen=200)

eye_closed_frames = 0
yawn_frames = 0

prev_time = time.time()

# ============================================
# MAIN LOOP
# ============================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    risk = 0
    status = "ALERT"
    color = (0, 255, 0)

    if results.multi_face_landmarks:

        lm = results.multi_face_landmarks[0].landmark

        # ========================================
        # FEATURE EXTRACTION
        # ========================================

        ear_val = (
            ear(lm, LEFT_EYE, w, h) +
            ear(lm, RIGHT_EYE, w, h)
        ) / 2

        mar_val = mar(lm, w, h)

        yaw_val = head_yaw(lm, w, h)

        tilt_val = head_tilt(lm, w, h)

        # ========================================
        # SMOOTHING
        # ========================================

        ear_hist.append(ear_val)
        mar_hist.append(mar_val)
        yaw_hist.append(yaw_val)
        tilt_hist.append(tilt_val)

        ear_avg = np.mean(ear_hist)
        mar_avg = np.mean(mar_hist)
        yaw_avg = np.mean(yaw_hist)
        tilt_avg = np.mean(tilt_hist)

        # ========================================
        # BLINK DETECTION
        # ========================================

        if ear_val < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:

            if 2 <= eye_closed_frames <= 8:
                blink_times.append(time.time())

            eye_closed_frames = 0

        # ========================================
        # YAWNING DETECTION
        # ========================================

        if mar_avg > MAR_THRESHOLD:
            yawn_frames += 1
        else:
            yawn_frames = 0

        is_yawning = yawn_frames > YAWN_FRAMES

        # ========================================
        # REMOVE OLD BLINKS
        # ========================================

        now = time.time()

        blink_times = deque(
            [t for t in blink_times if now - t <= BLINK_WINDOW_SEC],
            maxlen=200
        )

        blink_rate = len(blink_times)

        # ========================================
        # RISK SCORING
        # ========================================

        # Eye closure
        if ear_avg < EAR_THRESHOLD:
            risk += 2

        # Long eye closure
        if eye_closed_frames > 15:
            risk += 3

        # Low blink rate
        if blink_rate < 10:
            risk += 2

        # Yawning
        if is_yawning:
            risk += 3

        # Head turned away
        if abs(yaw_avg) > HEAD_YAW_THRESHOLD:
            risk += 1

        # Head tilt / nodding
        if abs(tilt_avg) > HEAD_TILT_THRESHOLD:
            risk += 2

        # ========================================
        # STATUS
        # ========================================

        if risk >= DROWSY_RISK:

            status = "DROWSY"
            color = (0, 0, 255)

        elif risk >= PRE_DROWSY_RISK:

            status = "PRE-DROWSY"
            color = (0, 165, 255)

        else:

            status = "ALERT"
            color = (0, 255, 0)

        # ========================================
        # DRAW LANDMARKS
        # ========================================

        important_points = (
            LEFT_EYE +
            RIGHT_EYE +
            MOUTH +
            [FOREHEAD, CHIN]
        )

        for i in important_points:

            x, y = pt(lm, i, w, h)

            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        # ========================================
        # TEXT
        # ========================================

        cv2.putText(
            frame,
            f"EAR: {ear_avg:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar_avg:.2f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw_avg:.2f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tilt: {tilt_avg:.1f}",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blinks/min: {blink_rate}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if is_yawning:

            cv2.putText(
                frame,
                "YAWNING DETECTED",
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ============================================
    # STATUS DISPLAY
    # ============================================

    cv2.putText(
        frame,
        status,
        (30, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        4
    )

    cv2.putText(
        frame,
        f"RISK: {risk}",
        (30, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3
    )

    # ============================================
    # FPS
    # ============================================

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ============================================
    # SHOW
    # ============================================

    cv2.imshow("Advanced Drowsiness Detection", frame)


    payload = {
        "ear": float(ear_avg),
        "mar": float(mar_avg),
        "yaw": float(yaw_avg),
        "tilt": float(tilt_avg),
        "blink_rate": float(blink_rate),
        "risk": int(risk),
        "status": status,
        "is_yawning": bool(is_yawning)
    }

    frame_count += 1

    if frame_count % SEND_EVERY_N_FRAMES == 0:
        send_to_server(payload)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()