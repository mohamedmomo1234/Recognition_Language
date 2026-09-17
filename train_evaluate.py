import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report
import joblib

print("[INFO] Loading dataset...")
df = pd.read_csv("dataset.csv")

df=df.dropna()

# فصل الـ Features عن الـ Labels
X = df.drop(columns=["label"])
y = df["label"]

# تقسيم البيانات 80% تدريب و 20% اختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تعريف الموديلات المقترحة
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=10),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

best_model_name = ""
best_model = None
best_accuracy = 0.0

comparsion_result=[]

print("\n[INFO] Training and evaluating models...")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    train_acc= accuracy_score(y_train, y_train_pred)
    test_acc= accuracy_score(y_test, y_test_pred)

    comparsion_result.append(
        {"Model": name, "Train Accracy": train_acc, "Test Accuracy": test_acc}
    )

    print(f"\n=================={name}=================")
    print(f"-> Train Accracy : {train_acc * 100:.2f}%")
    print(f"-> Test Accuracy (Accuracy) :{test_acc * 100:.2f}%")


    if train_acc - test_acc > 0.10:
     print(
        "[WARNING] Model might be suffering from Overfitting (High difference"
        " between train & test)."
    )
    else:
      print("[INFO] Generalization is good (No severe overfitting).")

    print("\nClassification Report:")
    
    print(classification_report(y_test, y_test_pred))

  # اختيار الأفضل بناءً على دقة الاختبار (Test Accuracy)
    if test_acc > best_accuracy:
     best_accuracy = test_acc
     best_model = model
     best_model_name = name

# طباعة جدول المقارنة النهائي
print("\n" + "=" * 40)
print("          MODEL COMPARISON TABLE          ")
print("=" * 40)
comparison_df = pd.DataFrame(comparsion_result)
print(comparison_df.to_string(index=False))
print("=" * 40)

print(
    f"\n[WINNER] Best Model is '{best_model_name}' with Test Accuracy:"
    f" {best_accuracy * 100:.2f}%"
)

# حفظ أفضل موديل
model_filename = "best_sign_model.pkl"
joblib.dump(best_model, model_filename)
print(f"[SUCCESS] Best model successfully saved to {model_filename}")