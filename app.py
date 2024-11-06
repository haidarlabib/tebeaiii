from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder 'uploads' jika belum ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
model = joblib.load('models/random_forest_model.pkl')

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman form
@app.route('/form')
def form():
    return render_template('form.html')

# Route untuk hasil prediksi
@app.route('/result', methods=['POST'])
def result():
    # Mengambil data dari form dan melakukan prediksi
    usia = int(request.form['usia'])
    jenis_kelamin = 1 if request.form['jenis_kelamin'] == 'Pria' else 0
    takaran_saji = int(request.form['takaran_saji'])
    jumlah_sajian = int(request.form['jumlah_sajian'])
    energi_total = int(request.form['energi_total'])
    energi_lemak = int(request.form['energi_lemak'])
    lemak_total = float(request.form['lemak_total'])
    lemak_jenuh = float(request.form['lemak_jenuh'])
    kolesterol = int(request.form['kolesterol'])
    karbohidrat = float(request.form['karbohidrat'])
    gula = float(request.form['gula'])
    garam = float(request.form['garam'])
    protein = float(request.form['protein'])

    input_data = np.array([[usia, jenis_kelamin, takaran_saji, jumlah_sajian, energi_total,
                            energi_lemak, lemak_total, lemak_jenuh, kolesterol, karbohidrat,
                            gula, garam, protein]])
    prediction = model.predict(input_data)
    prediction_result = 'Sehat' if prediction[0] == 1 else 'Tidak Sehat'
    return render_template('result.html', prediction_result=prediction_result)

# Route untuk halaman pengembangan
@app.route('/pengembangan')
def pengembangan():
    return render_template('pengembangan.html')

# Route untuk mengunggah file dan menghitung akurasi serta presisi
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Baca file dan hitung akurasi dan presisi
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return "File format not supported"

        # Pastikan kolom-kolom yang dibutuhkan ada
        required_columns = ["Usia", "Jenis Kelamin", "Takaran Saji (ml)", "Jumlah Sajian per Kemasan", 
                            "Energi Total (kcal)", "Energi dari Lemak (kcal)", "Lemak Total (g)", 
                            "Lemak Jenuh (g)", "Kolesterol (mg)", "Karbohidrat Total (g)", "Gula (g)", 
                            "Garam (g)", "Protein (g)", "Label Kesehatan"]
        
        if not all(column in df.columns for column in required_columns):
            return "File does not contain all required columns"

        # Encode 'Jenis Kelamin' and 'Label Kesehatan' and prepare X and y
        df["Jenis Kelamin"] = df["Jenis Kelamin"].map({"Pria": 1, "Wanita": 0})
        X = df[["Usia", "Jenis Kelamin", "Takaran Saji (ml)", "Jumlah Sajian per Kemasan", 
                "Energi Total (kcal)", "Energi dari Lemak (kcal)", "Lemak Total (g)", 
                "Lemak Jenuh (g)", "Kolesterol (mg)", "Karbohidrat Total (g)", "Gula (g)", 
                "Garam (g)", "Protein (g)"]]
        y_true = df["Label Kesehatan"].map({"Sehat": 1, "Tidak Sehat": 0})

        # Predict and calculate accuracy and precision
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        # Render hasil upload dan metrik
        return render_template('upload_result.html', filename=file.filename, accuracy=accuracy, precision=precision)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
