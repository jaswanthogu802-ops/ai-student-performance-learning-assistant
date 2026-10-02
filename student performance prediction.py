from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

print("Predicted marks:", round(prediction[0], 2))