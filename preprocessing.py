from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_columns = [
    "height", "weight", "ap_hi", "ap_lo", "calculated_age", "bmi"
]

categorical_columns = [
    "gender", "cholesterol", "gluc", "smoke", "alco", "active"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)
