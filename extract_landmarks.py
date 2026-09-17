import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# تهيئة MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)


DATASET_DIR = "Gesture Image Data/Gesture Image Data"  # ضع مسار مجلد الداتا ست هنا
data = []
labels = []

print("[INFO] Starting landmark extraction using (x, y) coordinates...")

for label in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    
    print(f"[INFO] Processing class: {label}")
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # تحويل الصورة إلى RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # استخراج إحداثيات x و y فقط (21 نقطة * 2 = 42 قيمة)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                landmarks = np.array(landmarks, dtype=np.float32)
                
                # --- Normalization (Translation & Scaling) ---
                # 1. Translation: جعل المعصم (Wrist - نقطة 0) هو نقطة الأصل (0,0)
                wrist = landmarks[0]
                landmarks = landmarks - wrist
                
                # 2. Scale Normalization: القسمة على أقصى مسافة لتوحيد الحجم
                max_distance = np.max(np.sqrt(np.sum(landmarks**2, axis=1)))
                if max_distance != 0:
                    landmarks = landmarks / max_distance
                
                # تحويل المصفوفة إلى شكل مسطح (Flat array of 42 features)
                flattened_features = landmarks.flatten().tolist()
                
                data.append(flattened_features)  
                labels.append(label)

hands.close()

# إنشاء DataFrame وحفظه في ملف CSV
columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}"])
columns.append("label")


dataset=[]
for features, label in zip(data, labels):
    dataset.append(features + [label])

df = pd.DataFrame(dataset, columns=columns)

df.to_csv("dataset.csv", index=False)
print(f"[SUCCESS] Extraction complete! Saved {len(df)} samples to dataset.csv")