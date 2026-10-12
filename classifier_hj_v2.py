import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 불러오기 (이미 train / validation / test로 나뉘어 있음)
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print("train:", train_df.shape, "validation:", val_df.shape, "test:", test_df.shape)
print(train_df["label"].value_counts())

X_train, y_train = train_df["text"], train_df["label"]
X_val, y_val = val_df["text"], val_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

# 2. 모델 만들고 학습 (train.csv만 사용)
model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

# 3. 검증 데이터로 평가 (설정을 바꿔 가며 비교할 때 이 점수를 봄)
val_pred = model.predict(X_val)
print("검증 정확도:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

# 4. 테스트 데이터로 최종 평가 (설정을 다 정한 뒤 마지막에 한 번만 봄)
test_pred = model.predict(X_test)
print("테스트 정확도:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# 5. 저장
joblib.dump(model, "classifier.joblib")

# 6. 새 문장으로 시험
new_text = ["택배가 아직 안 왔어요", "환불은 언제 되나요?", "비밀번호를 잊어버렸어요"]
print(model.predict(new_text))
