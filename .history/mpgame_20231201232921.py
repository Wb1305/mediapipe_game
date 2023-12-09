import cv2
import mediapipe as mp
import pydirectinput
from screeninfo import get_monitors  # thư viện lấy thông tin màn hình laptop

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # kích thước màn hình
        monitors = get_monitors()
        w = 1
        h = 1
        for monitor in monitors:
            w = monitor.width
            h = monitor.height
        # print(w, h)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks.landmark[9].x,
                #       hand_landmarks.landmark[9].y)
                pydirectinput.moveTo(
                    int((0.5-hand_landmarks.landmark[9].x) * w), int(hand_landmarks.landmark[9].y*h))
                if (hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y):
                    pydirectinput.mouseDown()
                else:
                    pydirectinput.mouseUp()
                if (hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x):
                    pydirectinput.click()
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # if cv2.waitKey(5) & 0xFF == 27:  # bấm nút esc để quit
        #     break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
cap.release()
