import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the student-mat.csv dataset
df = pd.read_csv('./dataset/student-mat.csv', delimiter=';')
featuresForPreprocessing = []
for column in df.columns:
    if isinstance(df[column][0], str):
        featuresForPreprocessing.append(column)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=featuresForPreprocessing)
# Convert target variables to binary values
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Select features and target variable
features = df.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
target = df['pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

LRmodel = LinearRegression()
LRmodel.fit(X_train, y_train)

predictonLR = LRmodel.predict(X_test)

DTCmodel = DecisionTreeClassifier(random_state=42)
DTCmodel.fit(X_train, y_train)

predictonDTC = DTCmodel.predict(X_test)

# Calculate MSE for both models
LRmse = mean_squared_error(y_test, predictonLR)
DTCmse = mean_squared_error(y_test, predictonDTC)

# Print the results
print("Linear Regression MSE:", LRmse)
print("Decision Tree Classifier MSE:", DTCmse)

