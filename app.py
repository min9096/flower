# app.py
import streamlit as st
import numpy as np
import joblib

# 모델 불러오기
model = joblib.load("model.pkl")

st.title("🌸 꽃 분류기 (Iris Classifier)")
st.write("입력값을 기반으로 꽃의 종류를 예측합니다.")

# 사용자 입력 받기
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 예측
prediction = model.predict(input_data)
predicted_class = prediction[0]
class_names = ['Setosa', 'Versicolor', 'Virginica']

st.subheader("🌼 예측 결과:")
st.write(f"→ {class_names[predicted_class]}")
