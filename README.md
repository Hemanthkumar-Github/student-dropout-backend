# 🎓 Student Dropout Risk Agent – Backend

This repository contains the backend service for the **Student Dropout Risk Agent**, an AI-powered system aligned with **SDG-4 (Quality Education)** to predict student dropout risk using machine learning and provide interactive chatbot-based assessment.

The backend is built using **FastAPI** and serves a trained **Random Forest model** to estimate dropout risk probability based on student academic and socio-economic inputs.

---

## 🚀 Features

- FastAPI-based REST backend  
- Interactive `/chat` API for step-by-step questioning  
- Machine Learning model for dropout prediction  
- Outputs:
  - Risk category (Low / Medium / High)
  - Dropout probability (%)  
- Supports deployment on **Render**  

---

## 🧠 Machine Learning Model

- Algorithm: Random Forest Classifier  
- Input Features:
  - Attendance percentage  
  - Average marks  
  - Family income level (1–5)  
  - Daily study hours  
  - Number of failed subjects  

- Output:
  - Dropout Risk (0 = No Dropout, 1 = Dropout)
  - Dropout Probability (%)

The trained model is stored as:

