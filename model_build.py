import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from preprocessing import preprocessor
from custom_logistic_regression import CustomLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("preprocessed_dataset.csv")

X = data.drop("cardio", axis=1)
y = data["cardio"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", CustomLogisticRegression(learning_rate=0.1, epochs=20000))
])

# pipeline.fit(X, y)



# to check accuracy
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Pipeline Accuracy:", acc)



joblib.dump(pipeline, "cardio_pipeline.pkl")

print("✅ Custom Logistic Regression Pipeline saved")
