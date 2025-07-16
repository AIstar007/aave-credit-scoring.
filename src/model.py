import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict_score(model, df_features):
    X = df_features.drop(columns=["userWallet"])
    return model.predict(X)