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

        monitors = get_monitors()
        screen_width = monitors[0].width
        screen_height = monitors[0].height
        # print(w, h)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks.landmark[9].x,
                #       hand_landmarks.landmark[9].y)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                pydirectinput.moveTo(
                    int((0.5-hand_landmarks.landmark[9].x) * w), int(hand_landmarks.landmark[9].y*h))
                if (hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y):
                    pydirectinput.mouseDown()
                else:
                    pydirectinput.mouseUp()
                if (hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x):
                    pydirectinput.click()
                 # Tính toán tọa độ và kích thước của khung chữ nhật bao quanh bàn tay
                min_x, min_y = screen_width, screen_height
                max_x, max_y = 0, 0
                for point in hand_landmarks.landmark:
                    x, y = int(
                        point.x * screen_width), int(point.y * screen_height)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

                # Hiển thị tên hành động trên khung chữ nhật
                action_label = "Your Action"
                cv2.rectangle(image, (min_x, min_y),
                              (max_x, max_y), (0, 255, 0), 2)
                cv2.putText(image, action_label, (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # if cv2.waitKey(5) & 0xFF == 27:  # bấm nút esc để quit
        #     break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()