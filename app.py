import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Raisin.csv')
# Chuyển đổi nhãn lớp từ dạng chuỗi sang số nguyên
data['Class'], class_names = pd.factorize(data['Class'])

# Chia dữ liệu thành features (đặc trưng) và labels (nhãn)
X = data.drop('Class', axis=1)
y = data['Class']

# Huấn luyện mô hình học máy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Lưu mô hình vào file pickle
pickle.dump(model, open('model.pkl', 'wb'))

# Khởi tạo Flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# Trang chủ của ứng dụng web
@app.route('/')
def home():
    return render_template('TrangChu.html')

@app.route('/Gioi_Thieu')
def Gioi_Thieu():
    return render_template('GioiThieu.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

# Dự đoán loại rượu dựa trên đầu vào từ người dùng
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Lấy giá trị từ các ô nhập liệu trong form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Dự đoán loại rượu dựa trên giá trị nhập vào
    prediction = model.predict(final_features)

    # Chuyển đổi kết quả dự đoán thành một số nguyên
    predicted_class_index = int(prediction[0])

    # Ánh xạ từ số liệu đầu ra sang tên loại rượu tương ứng
    class_mapping = dict(zip(range(len(class_names)), class_names))  # Sử dụng dictionary để ánh xạ lại tên nhãn lớp
    output = class_mapping[predicted_class_index]

    # Trả về kết quả dự đoán trên giao diện người dùng
    return render_template('TrangChu.html', prediction_text='Predicted Wine Type: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
