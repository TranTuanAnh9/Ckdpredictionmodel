from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ===== LOAD CÁC THỨ ĐÃ TRAIN =====
model = joblib.load("model.pkl")                 # log_clf
scaler = joblib.load("scaler.pkl")               # scaler đã fit
cols_to_scale = joblib.load("cols_to_scale.pkl") # các cột cần scale

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        conf_ckd=None,
        conf_notckd=None
    )

@app.route("/predict", methods=["POST"])
def predict():

    # ===== 1. LẤY DỮ LIỆU TỪ FORM =====
    age   = float(request.form["age"])
    bp    = float(request.form["bp"])
    sg    = float(request.form["sg"])
    al    = float(request.form["al"])
    su    = float(request.form["su"])
    rbc   = int(request.form["rbc"])
    pc    = int(request.form["pc"])
    htn   = int(request.form["htn"])
    dm    = int(request.form["dm"])
    appet = int(request.form["appet"])

    # ===== 2. TẠO DATAFRAME ĐÚNG NHƯ LÚC TEST =====
    X_df = pd.DataFrame([{
        'Tuổi': age,
        'Huyết Áp': bp,
        'Tỷ trọng nước tiểu': sg,
        'Hàm lượng albumin trong nước tiểu': al,
        'Mức đường trong nước tiểu': su,
        'Tình trạng hồng cầu': rbc,
        'Tình trạng bạch cầu mủ': pc,
        'Tăng huyết áp': htn,
        'Tiểu đường': dm,
        'Tình trạng ăn uống': appet
    }])

    # ===== 3. SCALE GIỐNG HỆT LÚC TRAIN =====
    X_df[cols_to_scale] = scaler.transform(X_df[cols_to_scale])

    # ===== 4. DỰ ĐOÁN =====
    pred = int(model.predict(X_df)[0])   # 0 = CKD, 1 = NOT CKD
    proba = model.predict_proba(X_df)[0]

    # ===== 5. ĐỘ TIN CẬY =====
    conf_ckd = round(proba[0] * 100, 2)
    conf_notckd = round(proba[1] * 100, 2)

    return render_template(
        "index.html",
        result=pred,
        conf_ckd=conf_ckd,
        conf_notckd=conf_notckd
    )

if __name__ == "__main__":
    app.run(debug=True)

