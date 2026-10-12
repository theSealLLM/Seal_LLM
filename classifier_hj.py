import pandas as pd

train = pd.read_csv("train.csv")

print(train.head())
print(train.shape)
print(train["label"].value_counts())

valid = pd.read_csv("validation.csv")

print(valid.head())
print(valid.shape)
print(valid["label"].value_counts())

from sklearn.model_selection import train_test_split

X_train = train["text"]
y_train = train["label"]
X_valid = valid["text"]
y_valid = valid["label"]


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

y_pred = model.predict(X_test)

print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "classifier.joblib")

import joblib

model =  joblib.load("classifier.joblib")

new_text = ["배송도 빠르고 픔질도 종아요", "포장이 찢어져 왔어요"]
print(model.predict(new_text))

