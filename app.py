import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Title and Description
st.write("## Personal Fitness Tracker")
st.write("Fill out the form below to track your fitness, predict calories burned, and get personalized suggestions to improve your health.")

# Form for User Input
st.subheader("Fitness Tracking Form")
with st.form(key="fitness_form"):
    st.write("**Enter Your Details**")
    age = st.number_input("Age", min_value=20, max_value=80, value=30, step=1, help="Age range: 20-80")
    weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.1, help="Your weight in kilograms")
    height = st.number_input("Height (cm)", min_value=140.0, max_value=220.0, value=175.0, step=0.1, help="Your height in centimeters")
    duration = st.number_input("Duration (min)", min_value=0, max_value=30, value=20, step=1, help="Exercise duration in minutes")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=130, value=84, step=1, help="Average heart rate during exercise")
    body_temp = st.number_input("Body Temp (°C)", min_value=36.0, max_value=42.0, value=36.9, step=0.1, help="Body temperature during exercise")
    gender = st.radio("Gender", ("Male", "Female"), help="Select your gender")
    activity_type = st.selectbox("Activity Type", ("Walking", "Running", "Cycling", "Strength Training"), help="Type of exercise performed")

    # Submit Button
    submit_button = st.form_submit_button(label="Track Fitness")

# Process Form Submission
if submit_button:
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)

    # Encode gender and activity type
    gender_encoded = 1 if gender == "Male" else 0
    activity_mapping = {"Walking": 1, "Running": 2, "Cycling": 3, "Strength Training": 4}
    activity_encoded = activity_mapping[activity_type]

    # Create DataFrame for prediction
    user_data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_encoded,
        "Activity_Type": activity_encoded
    }
    df_input = pd.DataFrame(user_data, index=[0])

    # Display User Input
    st.write("---")
    st.subheader("Your Submitted Details")
    user_display = df_input.copy()
    user_display["Gender_male"] = "Male" if gender_encoded == 1 else "Female"
    user_display["Activity_Type"] = activity_type
    st.write(user_display)

    # Load and Preprocess Data
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
    except FileNotFoundError:
        st.error("Error: 'calories.csv' or 'exercise.csv' not found. Please add them to the project folder.")
        st.stop()

    # Merge datasets and drop User_ID
    exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

    # Add BMI and age groups
    for data in [exercise_df]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = data["BMI"].round(2)
        data["age_groups"] = pd.cut(data["Age"], bins=[20, 40, 60, 80], labels=["Young", "Middle-Aged", "Old"], right=False)

    # Split into train and test sets
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Add BMI and age groups to train/test sets
    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = data["BMI"].round(2)
        data["age_groups"] = pd.cut(data["Age"], bins=[20, 40, 60, 80], labels=["Young", "Middle-Aged", "Old"], right=False)

    # Simulate Activity Type (since dataset doesn't have it, we'll add a dummy column for training)
    np.random.seed(1)
    exercise_train_data["Activity_Type"] = np.random.choice([1, 2, 3, 4], size=len(exercise_train_data))
    exercise_test_data["Activity_Type"] = np.random.choice([1, 2, 3, 4], size=len(exercise_test_data))

    # Select relevant columns and encode Gender
    columns = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Activity_Type", "Calories"]
    exercise_train_data = exercise_train_data[columns]
    exercise_test_data = exercise_test_data[columns]
    exercise_train_data = pd.get_dummies(exercise_train_data, columns=["Gender"], drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, columns=["Gender"], drop_first=True)

    # Prepare features and target
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    # Train RandomForestRegressor
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
    random_reg.fit(X_train, y_train)

    # Evaluate Model
    y_pred_test = random_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    # Align user input with training columns
    df_input = df_input.reindex(columns=X_train.columns, fill_value=0)

    # Make Prediction
    prediction = random_reg.predict(df_input)

    # Display Prediction
    st.write("---")
    st.subheader("Predicted Calories Burned")
    st.write(f"{prediction[0]:.2f} **kilocalories**")
    st.write(f"*Model RMSE: {rmse:.2f}, MAE: {mae:.2f} (based on test data)*")

    # Comparison with Dataset
    st.write("---")
    st.subheader("How You Compare")
    age_percent = (exercise_df["Age"] < age).mean() * 100
    duration_percent = (exercise_df["Duration"] < duration).mean() * 100
    heart_rate_percent = (exercise_df["Heart_Rate"] < heart_rate).mean() * 100
    body_temp_percent = (exercise_df["Body_Temp"] < body_temp).mean() * 100
    bmi_percent = (exercise_df["BMI"] < bmi).mean() * 100

    st.write(f"You are older than {age_percent:.1f}% of people in the dataset.")
    st.write(f"Your exercise duration is higher than {duration_percent:.1f}% of people.")
    st.write(f"Your heart rate is higher than {heart_rate_percent:.1f}% of people during exercise.")
    st.write(f"Your body temperature is higher than {body_temp_percent:.1f}% of people during exercise.")
    st.write(f"Your BMI is higher than {bmi_percent:.1f}% of people in the dataset.")

    # Fitness Improvement Suggestions
    st.write("---")
    st.subheader("Fitness Improvement Suggestions")

    # BMI Category
    bmi_category = "Unknown"
    if bmi < 15:
        bmi_category = "Very severely underweight"
    elif 15 <= bmi < 16:
        bmi_category = "Severely underweight"
    elif 16 <= bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
    elif 30 <= bmi < 35:
        bmi_category = "Obese Class I"
    elif 35 <= bmi < 40:
        bmi_category = "Obese Class II"
    else:
        bmi_category = "Obese Class III"

    st.write(f"**Your BMI Category**: {bmi_category}")
    if bmi < 18.5:
        st.write("- **BMI Suggestion**: Your BMI is below the healthy range. Focus on a balanced diet with more protein and healthy fats. Consider strength training to build muscle mass.")
    elif 18.5 <= bmi < 25:
        st.write("- **BMI Suggestion**: Your BMI is in the healthy range. Maintain your current fitness routine and ensure a balanced diet with adequate nutrients.")
    elif 25 <= bmi < 30:
        st.write("- **BMI Suggestion**: Your BMI indicates you are overweight. Aim for a calorie deficit by increasing exercise duration or intensity and reducing processed food intake.")
    else:
        st.write("- **BMI Suggestion**: Your BMI indicates obesity. Consult a healthcare professional for a tailored weight loss plan. Start with low-impact exercises like walking or swimming.")

    # Duration Suggestion
    median_duration = exercise_df["Duration"].median()
    if duration < median_duration:
        st.write(f"- **Duration Tip**: Your exercise duration ({duration} min) is below the dataset median ({median_duration} min). Try increasing it by 5-10 minutes to burn more calories.")
    else:
        st.write(f"- **Duration Tip**: Your exercise duration ({duration} min) is above the dataset median ({median_duration} min). Great job! Consider adding variety to your workouts, such as interval training.")

    # Heart Rate Suggestion
    max_hr = 220 - age  # Approximate max heart rate
    target_hr_low = max_hr * 0.5
    target_hr_high = max_hr * 0.85
    if heart_rate < target_hr_low:
        st.write(f"- **Heart Rate Tip**: Your heart rate ({heart_rate} bpm) is below the target zone ({target_hr_low:.0f}-{target_hr_high:.0f} bpm) for your age. Increase workout intensity to improve cardiovascular fitness.")
    elif heart_rate > target_hr_high:
        st.write(f"- **Heart Rate Tip**: Your heart rate ({heart_rate} bpm) is above the target zone ({target_hr_low:.0f}-{target_hr_high:.0f} bpm). Avoid overexertion; consider reducing intensity or consulting a doctor.")
    else:
        st.write(f"- **Heart Rate Tip**: Your heart rate ({heart_rate} bpm) is within the target zone ({target_hr_low:.0f}-{target_hr_high:.0f} bpm) for your age. Great job! This is optimal for fat burning and cardiovascular health.")

    # Activity Type Suggestion
    st.write(f"**Activity Type**: {activity_type}")
    if activity_type in ["Walking", "Cycling"]:
        st.write("- **Activity Tip**: Low-impact activities like walking or cycling are great for endurance. To burn more calories, try adding intervals (e.g., 1 min fast, 2 min slow).")
    elif activity_type == "Running":
        st.write("- **Activity Tip**: Running is excellent for calorie burning. Ensure proper footwear to avoid injury, and consider cross-training with strength exercises.")
    else:  # Strength Training
        st.write("- **Activity Tip**: Strength training builds muscle, which boosts metabolism. Combine with cardio sessions to maximize calorie burn.")

    # Age Group Insights
    age_group = "Young" if age < 40 else "Middle-Aged" if age < 60 else "Old"
    st.write(f"**Your Age Group**: {age_group}")
    age_group_data = exercise_df[exercise_df["age_groups"] == age_group]
    median_calories = age_group_data["Calories"].median()
    st.write(f"- **Age Group Insight**: The median calories burned for your age group ({age_group}) is {median_calories:.1f} kcal. You burned {prediction[0]:.1f} kcal, which is {'above' if prediction[0] > median_calories else 'below'} the median.")

    # Visual Insights
    st.write("---")
    st.subheader("Your Position in the Dataset")
    fig_bmi = px.histogram(exercise_df, x="BMI", nbins=30, title="BMI Distribution in Dataset")
    fig_bmi.add_vline(x=bmi, line_dash="dash", line_color="red", annotation_text="Your BMI")
    st.plotly_chart(fig_bmi, use_container_width=True)

    st.write("---")
    st.subheader("Calories Burned by Age Group")
    fig_age = px.box(exercise_df, x="age_groups", y="Calories", color="Gender", title="Calories Burned by Age Group and Gender")
    st.plotly_chart(fig_age, use_container_width=True)

    st.write("---")
    st.subheader("Key Insight: Duration and Calories")
    st.write("The dataset shows a strong correlation between exercise duration and calories burned. Increasing your duration can significantly boost your calorie burn.")
    fig_corr = px.scatter(exercise_df, x="Duration", y="Calories", color="Gender", size="Heart_Rate", title="Duration vs Calories Burned")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Additional Health Tips
    st.write("---")
    st.subheader("General Health Tips")
    st.write("- **Hydration**: Drink at least 2-3 liters of water daily, especially after exercise.")
    st.write("- **Nutrition**: Include protein-rich foods (e.g., chicken, eggs, lentils) to support muscle recovery.")
    st.write("- **Rest**: Ensure 7-8 hours of sleep to aid recovery and improve performance.")