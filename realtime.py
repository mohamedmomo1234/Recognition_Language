import cv2
import mediapipe as mp
import numpy as np
import joblib

# تحميل الموديل المدرب مسبقاً
model = joblib.load("best_sign_model.pkl")
print("[INFO] Model loaded successfully for Real-Time Prediction.")

# تهيئة MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# فتح الويب كام
cap = cv2.VideoCapture(0)

print("[INFO] Starting Webcam... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # انعكاس الصورة لتبدو كمرآة، وتحويل الألوان لـ RGB
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    predicted_label = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # رسم نقاط اليد على الشاشة
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # استخراج إحداثيات x و y فقط (42 قيمة)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # نفس خطوات الـ Normalization تماماً كما في ملف التدريب
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            
            max_distance = np.max(np.sqrt(np.sum(landmarks**2, axis=1)))
            if max_distance != 0:
                landmarks = landmarks / max_distance
                
            flattened_features = landmarks.flatten().reshape(1, -1)
            
            # التوقع باستخدام الموديل المدرب
            prediction = model.predict(flattened_features)
            predicted_label = str(prediction[0])

    # عرض النتيجة على الشاشة
    cv2.putText(
        frame, 
        f"Prediction: {predicted_label}", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.2, 
        (0, 255, 0), 
        3, 
        cv2.LINE_AA
    )

    cv2.imshow("Sign Language Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()