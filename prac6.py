import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
seed = 7
num_trees = 30

# Create the AdaBoost model using the SAMME algorithm
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed, algorithm='SAMME')

# Perform cross-validation and calculate the mean accuracy
results = model_selection.cross_val_score(model, X, Y)
print(results.mean())
