from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # demo/ hoặc app/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # thư mục gốc project
MODEL_DIR = os.path.join(ROOT_DIR, "model2xacnhan")     # ✅ ĐÚNG TÊN THƯ MỤC

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates")
)

# ===== LOAD MODEL & TOOLS =====
model = joblib.load(os.path.join(MODEL_DIR, "model2xacnhan.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler2xacnhan.pkl"))
cols_to_scale = joblib.load(os.path.join(MODEL_DIR, "cols_to_scale2xacnhan.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders2xacnhan.pkl"))

FEATURE_ORDER = [
    'Tuổi','Huyết Áp','Tỷ trọng nước tiểu',
    'Hàm lượng albumin trong nước tiểu','Mức đường trong nước tiểu',
    'Tình trạng hồng cầu','Tình trạng bạch cầu mủ',
    'Cụm bạch cầu mủ','Vi khuẩn trong nước tiểu',
    'Đường huyết ngẫu nhiên','Ure máu','Creatinine huyết thanh',
    'Natri','kali','Huyết sắc tố','Thể tích hồng cầu đóng gói',
    'Số lượng bạch cầu','Số lượng hồng cầu',
    'Tăng huyết áp','Tiểu đường','Bệnh động mạch vành',
    'Tình trạng ăn uống','Phù chân','Thiếu máu'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index2xacnhan.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():

    def f(name):
        v = request.form.get(name)
        return float(v) if v not in ("", None) else np.nan

    def i(name):
        return int(request.form[name])

    X_df = pd.DataFrame([{
        'Tuổi': f('age'),
        'Huyết Áp': f('bp'),
        'Tỷ trọng nước tiểu': f('sg'),
        'Hàm lượng albumin trong nước tiểu': f('al'),
        'Mức đường trong nước tiểu': f('su'),

        'Tình trạng hồng cầu': i('rbc'),
        'Tình trạng bạch cầu mủ': i('pc'),
        'Cụm bạch cầu mủ': i('pcc'),
        'Vi khuẩn trong nước tiểu': i('ba'),

        'Đường huyết ngẫu nhiên': f('rbg'),
        'Ure máu': f('bu'),
        'Creatinine huyết thanh': f('sc'),
        'Natri': f('sod'),
        'kali': f('pot'),
        'Huyết sắc tố': f('hemo'),
        'Thể tích hồng cầu đóng gói': f('pcv'),
        'Số lượng bạch cầu': f('wc'),
        'Số lượng hồng cầu': f('rc'),

        'Tăng huyết áp': i('htn'),
        'Tiểu đường': i('dm'),
        'Bệnh động mạch vành': i('cad'),
        'Tình trạng ăn uống': i('appet'),
        'Phù chân': i('pe'),
        'Thiếu máu': i('ane')
    }])[FEATURE_ORDER]

    # ===== SCALE =====
    X_df[cols_to_scale] = scaler.transform(X_df[cols_to_scale])

    pred = int(model.predict(X_df)[0])
    proba = model.predict_proba(X_df)[0]

    return render_template(
        "index2xacnhan.html",
        result=pred,
        conf_ckd=round(proba[0]*100, 2),
        conf_notckd=round(proba[1]*100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
