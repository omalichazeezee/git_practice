import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# load the data
dataset = pd.read_csv("iris.csv")
print(dataset.head())

# select dependent and independent variables
X = dataset[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = dataset["Class"]

# split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# instantiate the model
classifier = RandomForestClassifier()

# fit the model
classifier.fit(X_train, y_train)

# make pickle file of the model
pickle.dump(classifier, open("model.pkl", "wb"))
