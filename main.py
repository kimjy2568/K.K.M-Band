import cv2
import mediapipe as mp
import winsound

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 음계와 대응하는 주파수 정의 (도, 레, 미, 파, 솔, 라, 시)
notes = {
    "thumb_folded": {"note": "도", "frequency": 261},
    "index_folded": {"note": "레", "frequency": 293},
    "middle_folded": {"note": "미", "frequency": 329},
    "ring_folded": {"note": "파", "frequency": 349},
    "pinky_folded": {"note": "솔", "frequency": 392},
    "only_thumb_stretched": {"note": "라", "frequency": 440},
    "only_index_stretched": {"note": "시", "frequency": 493}
}

def play_sound(note_info):
    frequency = note_info["frequency"]
    note = note_info["note"]
    duration = 500  # 500ms
    winsound.Beep(frequency, duration)
    print(f"Playing sound: {note}")

def is_finger_folded(hand_landmarks, finger_tip, finger_dip):
    return hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_dip].y

# 웹캠 설정
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe 프레임 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손가락 상태 확인
            thumb_folded = is_finger_folded(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP)
            index_folded = is_finger_folded(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP)
            middle_folded = is_finger_folded(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
            ring_folded = is_finger_folded(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP)
            pinky_folded = is_finger_folded(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)

            if thumb_folded and not index_folded and not middle_folded and not ring_folded and not pinky_folded:
                play_sound(notes["thumb_folded"])
            elif index_folded and not thumb_folded and not middle_folded and not ring_folded and not pinky_folded:
                play_sound(notes["index_folded"])
            elif middle_folded and not thumb_folded and not index_folded and not ring_folded and not pinky_folded:
                play_sound(notes["middle_folded"])
            elif ring_folded and not thumb_folded and not index_folded and not middle_folded and not pinky_folded:
                play_sound(notes["ring_folded"])
            elif pinky_folded and not thumb_folded and not index_folded and not middle_folded and not ring_folded:
                play_sound(notes["pinky_folded"])
            elif not thumb_folded and not index_folded and middle_folded and ring_folded and not pinky_folded:
                play_sound(notes["only_thumb_stretched"])
            elif not thumb_folded and not index_folded and not middle_folded and ring_folded and pinky_folded:
                play_sound(notes["only_index_stretched"])

    # 프레임 표시
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()
