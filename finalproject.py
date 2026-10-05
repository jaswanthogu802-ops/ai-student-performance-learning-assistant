# ============================================================
# AI-BASED STUDENT PERFORMANCE & LEARNING ASSISTANT
# IBM SkillsBuild Final Project
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

import gymnasium as gym


# ============================================================
# 1. STUDENT DATASET
# ============================================================

data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 4,
        6, 8, 3, 7, 5, 9
    ],

    "previous_marks": [
        35, 40, 45, 50, 55, 60, 65, 70, 78, 85,
        42, 48, 52, 58, 63, 68, 73, 80, 88, 47,
        62, 75, 44, 69, 57, 82
    ],

    "attendance": [
        65, 70, 72, 75, 78, 80, 82, 85, 88, 92,
        68, 74, 76, 79, 81, 83, 86, 90, 94, 73,
        82, 87, 70, 84, 77, 91
    ],

    "assignment_score": [
        40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
        48, 52, 58, 63, 68, 73, 78, 85, 92, 55,
        67, 79, 49, 72, 61, 87
    ],

    "final_marks": [
        38, 43, 48, 53, 59, 64, 69, 75, 82, 90,
        45, 50, 55, 61, 66, 72, 77, 84, 93, 51,
        67, 79, 47, 71, 60, 86
    ]
}


df = pd.DataFrame(data)


print("\n================================================")
print(" AI-BASED STUDENT PERFORMANCE & LEARNING ASSISTANT")
print("================================================")


print("\nDataset loaded successfully!")

print("Total Students:", len(df))


# ============================================================
# 2. SUPERVISED LEARNING
# Linear Regression
# ============================================================

print("\n\n================ SUPERVISED LEARNING ================")

X = df[
    [
        "study_hours",
        "previous_marks",
        "attendance",
        "assignment_score"
    ]
]

y = df["final_marks"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model

linear_model = LinearRegression()


# Train model

linear_model.fit(X_train, y_train)


# Test model

predictions = linear_model.predict(X_test)


# Model evaluation

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("Model: Linear Regression")

print("Mean Absolute Error:",
      round(mae, 2))

print("R2 Score:",
      round(r2, 2))


# ============================================================
# 3. UNSUPERVISED LEARNING
# K-Means Clustering
# ============================================================

print("\n\n================ UNSUPERVISED LEARNING ================")


cluster_features = df[
    [
        "study_hours",
        "previous_marks",
        "attendance",
        "assignment_score"
    ]
]


# Create K-Means model

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Create clusters

df["cluster"] = kmeans.fit_predict(
    cluster_features
)


print("Algorithm: K-Means")

print("Number of Clusters: 3")


print("\nCluster Distribution:")

print(
    df["cluster"].value_counts().sort_index()
)


# ============================================================
# 4. REINFORCEMENT LEARNING
# LunarLander Environment
# ============================================================

print("\n\n================ REINFORCEMENT LEARNING ================")


env = gym.make(
    "LunarLander-v3"
)


state_size = env.observation_space.shape[0]

action_size = env.action_space.n


print("Environment: LunarLander-v3")

print("State Size:", state_size)

print("Action Size:", action_size)


# Reset environment

state, info = env.reset(
    seed=42
)


print("\nInitial State:")

print(state)


# Random action demonstration

action = env.action_space.sample()


next_state, reward, terminated, truncated, info = env.step(
    action
)


print("\nAction:", action)

print("Reward:", round(reward, 2))

print("Next State:", next_state)

print("Terminated:", terminated)

print("Truncated:", truncated)


env.close()


# ============================================================
# 5. DQN NETWORK
# Demonstration of Deep Q Learning architecture
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


print("\n\n================ DQN NETWORK ================")


class DQN(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super(DQN, self).__init__()


        self.fc1 = nn.Linear(
            state_size,
            64
        )


        self.fc2 = nn.Linear(
            64,
            64
        )


        self.fc3 = nn.Linear(
            64,
            action_size
        )


    def forward(self, state):

        x = self.fc1(state)

        x = F.relu(x)


        x = self.fc2(x)

        x = F.relu(x)


        return self.fc3(x)


# Create DQN

dqn_model = DQN(
    state_size,
    action_size
)


print("DQN Network Created Successfully!")

print(dqn_model)


# ============================================================
# 6. USER INPUT
# ============================================================

print("\n\n================ STUDENT ANALYSIS ================")


study_hours = float(
    input("\nEnter daily study hours: ")
)


previous_marks = float(
    input("Enter previous marks: ")
)


attendance = float(
    input("Enter attendance percentage: ")
)


assignment_score = float(
    input("Enter assignment score: ")
)


# ============================================================
# 7. PREDICT FINAL MARKS
# ============================================================

student_data = np.array([
    [
        study_hours,
        previous_marks,
        attendance,
        assignment_score
    ]
])


predicted_marks = linear_model.predict(
    student_data
)[0]


# Keep marks between 0 and 100

predicted_marks = max(
    0,
    min(100, predicted_marks)
)


# ============================================================
# 8. PERFORMANCE CATEGORY
# ============================================================

if predicted_marks >= 75:

    performance = "Excellent"

elif predicted_marks >= 60:

    performance = "Good"

elif predicted_marks >= 40:

    performance = "Average"

else:

    performance = "Needs Improvement"


# ============================================================
# 9. FIND STUDENT CLUSTER
# ============================================================

student_cluster = kmeans.predict(
    student_data
)[0]


# ============================================================
# 10. AI LEARNING RECOMMENDATIONS
# ============================================================

recommendations = []


if study_hours < 4:

    recommendations.append(
        "Increase your daily study time to at least 4 hours."
    )


if previous_marks < 50:

    recommendations.append(
        "Revise previous topics and practice basic questions."
    )


if attendance < 75:

    recommendations.append(
        "Improve your attendance and attend classes regularly."
    )


if assignment_score < 60:

    recommendations.append(
        "Spend more time completing assignments and practice problems."
    )


if predicted_marks >= 75:

    recommendations.append(
        "Excellent performance. Continue your current learning routine."
    )

elif predicted_marks >= 60:

    recommendations.append(
        "Good performance. Focus on weak subjects to improve further."
    )

elif predicted_marks >= 40:

    recommendations.append(
        "Your performance is average. Increase revision and practice."
    )

else:

    recommendations.append(
        "Your performance needs improvement. Follow a daily study plan."
    )


# ============================================================
# 11. FINAL STUDENT REPORT
# ============================================================

print("\n\n================================================")
print("              FINAL STUDENT REPORT")
print("================================================")


print("\nStudy Hours:",
      study_hours)


print("Previous Marks:",
      previous_marks)


print("Attendance:",
      attendance, "%")


print("Assignment Score:",
      assignment_score)


print("\nPredicted Final Marks:",
      round(predicted_marks, 2))


print("Performance:",
      performance)


print("Student Cluster:",
      student_cluster)


# ============================================================
# 12. RECOMMENDATIONS
# ============================================================

print("\n\n================ AI RECOMMENDATIONS ================")


for number, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{number}. {recommendation}"
    )


# ============================================================
# 13. PROJECT SUMMARY
# ============================================================

print("\n\n================================================")
print("                 PROJECT SUMMARY")
print("================================================")


print("""
This project demonstrates multiple AI and Machine
Learning concepts learned through IBM SkillsBuild
masterclasses.

1. Supervised Learning
   - Linear Regression
   - Student final marks prediction

2. Unsupervised Learning
   - K-Means Clustering
   - Student learning group identification

3. Reinforcement Learning
   - LunarLander environment
   - State, action and reward concepts

4. Deep Q Learning
   - Neural Network architecture
   - State-to-action value prediction

5. AI Learning Assistant
   - Personalized learning recommendations

The system helps identify student performance,
predict future marks and provide learning guidance.
""")


print("\nProject executed successfully!")
