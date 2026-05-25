import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\Administrador\Downloads\Bengaluru_House_Data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def _convert_sqft(x):
    tokens = str(x).split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except ValueError:
        return None


def _remove_pps_outliers(df):
    frames = []
    for _, subdf in df.groupby("location"):
        m = subdf["price_per_sqft"].mean()
        s = subdf["price_per_sqft"].std()
        frames.append(subdf[(subdf["price_per_sqft"] > m - s) & (subdf["price_per_sqft"] <= m + s)])
    return pd.concat(frames, ignore_index=True)


def preprocess(df):
    df = df.copy()

    df.drop(columns=["society"], inplace=True)

    df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) else None)
    df.drop(columns=["size"], inplace=True)

    df["total_sqft"] = df["total_sqft"].apply(_convert_sqft)

    df.dropna(subset=["total_sqft", "bath", "bhk", "price", "location"], inplace=True)
    df["balcony"] = df["balcony"].fillna(df["balcony"].median())

    df["location"] = df["location"].str.strip()
    loc_counts = df["location"].value_counts()
    df["location"] = df["location"].apply(lambda x: x if loc_counts.get(x, 0) >= 10 else "other")

    df["price_per_sqft"] = df["price"] * 100_000 / df["total_sqft"]

    df = df[df["total_sqft"] / df["bhk"] >= 300]
    df = _remove_pps_outliers(df)
    df = df[df["bath"] < df["bhk"] + 2]

    return df.reset_index(drop=True)


def build_features(df):
    dummies = pd.get_dummies(df["location"])
    X = pd.concat([df[["total_sqft", "bath", "bhk"]], dummies], axis=1)
    y = df["price"]
    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models = {
        "Regressão Linear": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0),
        "Lasso (L1)": Lasso(alpha=1.0, max_iter=10_000),
    }

    results = {}
    trained = {}

    for name, base_model in base_models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results[name] = {
            "R²": round(r2_score(y_test, y_pred), 4),
            "MAE (Lakhs)": round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE (Lakhs)": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "CV R² (5-fold)": round(cross_val_score(pipe, X, y, cv=5, scoring="r2").mean(), 4),
        }
        trained[name] = (pipe, y_test.values, y_pred)

    return results, trained, X.columns.tolist()


def get_pvalues(X, y):
    import statsmodels.api as sm
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_sm = sm.add_constant(X_scaled)
    ols = sm.OLS(y, X_sm).fit()
    coefs = pd.Series(ols.params[1:].values, index=X.columns)
    pvals = pd.Series(ols.pvalues[1:].values, index=X.columns)
    return coefs, pvals


def predict_price(pipeline, X_columns, location, sqft, bath, bhk):
    x = np.zeros(len(X_columns))
    x[X_columns.index("total_sqft")] = sqft
    x[X_columns.index("bath")] = bath
    x[X_columns.index("bhk")] = bhk
    if location in X_columns:
        x[X_columns.index(location)] = 1
    return max(0.0, pipeline.predict([x])[0])
