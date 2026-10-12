import pandas as pd

train_df = pd.read_csv("data/train.csv")

print(train_df.head())
print(train_df.shape)
print(train_df["label"].value_counts())

val_df = pd.read_csv("data/validation.csv")

print(val_df.head())
print(val_df.shape)
print(val_df["label"].value_counts())

test_df = pd.read_csv("data/test.csv")

print(test_df.head())
print(test_df.shape)
print(test_df["label"].value_counts())

from sklearn.model_selection import train_test_split

X_train = train_df["text"]
y_train = train_df["label"]
X_val = val_df["text"]
y_val = val_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]



# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

val_pred = model.predict(X_val)

print("검증 정확도:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

test_pred = model.predict(X_test)

print("테스트 정확도:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

import joblib

joblib.dump(model, "classifier.joblib")

import joblib

model =  joblib.load("classifier.joblib")

new_text = ["배송도 빠르고 픔질도 종아요", "포장이 찢어져 왔어요"]
print(model.predict(new_text))

