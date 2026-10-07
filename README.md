AI Student Performance & Learning Assistant

An AI-based student performance analysis system that predicts final marks, groups students based on learning patterns, and provides personalized learning recommendations.

Features

- Student final marks prediction using Linear Regression
- Student grouping using K-Means Clustering
- Reinforcement Learning using LunarLander-v3
- DQN neural network implementation using PyTorch
- Performance classification
- Personalized AI learning recommendations
- Student performance analysis using study hours, previous marks, attendance, and assignment scores

Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Gymnasium
- PyTorch
- Linear Regression
- K-Means Clustering
- Deep Q Network (DQN)

Machine Learning Concepts

1. Supervised Learning

Linear Regression is used to predict a student's final marks based on:

- Study hours
- Previous marks
- Attendance
- Assignment score

2. Unsupervised Learning

K-Means Clustering is used to group students into three learning-performance clusters.

3. Reinforcement Learning

The Gymnasium LunarLander-v3 environment demonstrates states, actions, rewards, terminated states, and truncated states.

4. Deep Learning

A DQN neural network is created using PyTorch to map environment states to action values.

How to Run

Install the required libraries:

pip install numpy pandas scikit-learn gymnasium torch

Run the project:

python "your_file_name.py"

Enter the student's:

- Daily study hours
- Previous marks
- Attendance percentage
- Assignment score

The system generates a final report containing predicted marks, performance level, student cluster, and personalized recommendations.

Project Output

The system provides:

- Predicted Final Marks
- Performance Level
- Student Cluster
- AI-based Learning Recommendations

Project Purpose

This project demonstrates the practical application of AI and Machine Learning concepts learned through IBM SkillsBuild masterclasses and combines multiple AI techniques into a single student learning assistant.

Author

Jaswanth Ogu

BTech CSE Student | Aspiring Full Stack Developer & AI/ML Enthusiast
