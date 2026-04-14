import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------- APP START ----------------
st.write("App started")

# ---------------- MODEL LOAD ----------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.title("🎓 Student Exam Score Predictor")
st.write("Enter student details:")

# ---------------- INPUTS ----------------
age = st.number_input("Age", 10, 30, 20)
gender = st.selectbox("Gender", ["male", "female", "other"])
course = st.selectbox("Course", ["b.sc", "b.com", "bca", "b.tech", "bba", "diploma", "ba"])
study_hours = st.number_input("Study Hours", 0, 12, 5)
class_attendance = st.slider("Class Attendance (%)", 0, 100, 75)
internet_access = st.selectbox("Internet Access", ["yes", "no"])
sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)
sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
study_method = st.selectbox("Study Method", ["coaching", "online videos", "group study", "mixed", "self-study"])
facility_rating = st.slider("Facility Rating", 1, 5, 3)
exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])

# ---------------- PREDICTION ----------------
if st.button("Predict Score"):

    sample = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    try:
        prediction = model.predict(sample)[0]
        st.success(f"🎯 Predicted Exam Score: {round(prediction, 2)}")

        # ---------------- INSIGHTS ----------------
        st.subheader("📊 Insights")

        if study_hours <= 2:
            st.write("⚠️ Very low study hours")
        elif study_hours <= 5:
            st.write("📘 Moderate study hours")
        else:
            st.write("🔥 Good study hours")

        if class_attendance < 50:
            st.write("❌ Low attendance")
        elif class_attendance < 75:
            st.write("⚠️ Average attendance")
        else:
            st.write("✅ Good attendance")

        if sleep_quality == "poor":
            st.write("😴 Poor sleep quality")
        elif sleep_quality == "good":
            st.write("😌 Good sleep quality")

        if internet_access == "no":
            st.write("🌐 No internet access may limit learning")

        # ---------------- VISUALIZATION ----------------
        st.subheader("📊 Performance Overview")

        labels = ["Study Hours", "Attendance (%)", "Sleep Hours"]

        values = [
            study_hours,
            class_attendance,
            sleep_hours
        ]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        ax.set_ylabel("Value")
        ax.set_title("Student Performance Comparison")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")