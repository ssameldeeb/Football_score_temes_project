import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Football teams.csv")

print(data.shape)
print(data.size)
print(data.columns.values)
print(len(data.columns.values))
print(data.isnull().sum())
print(data.dtypes)

La = LabelEncoder()

data["Team"] = La.fit_transform(data["Team"])
data["Tournament"] = La.fit_transform(data["Tournament"])


print(data.dtypes)

plt.figure(figsize=(9,6))
sns.heatmap(data.corr(), annot=True)
plt.show()

data["Pass"] = data["Pass%"]
data = data.drop("Pass%",axis=1)

data["Possession"] = data["Possession%"]
data = data.drop("Possession%",axis=1)

print(data.dtypes)

x = data.drop("Rating",axis=1)
y = data["Rating"]
print(x.shape)
print(y.shape)
print(y.unique())

mm = MinMaxScaler(copy=True, feature_range=(0, 5))
x = mm.fit_transform(x)
print(x[:5])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# models
Lo = LinearRegression()
Lo.fit(X_train, y_train)

print("_"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))

y_pred = Lo.predict(X_test)
print(y_test[:5])
print(y_pred[:5])

ma = mean_absolute_error(y_test,y_pred)
print(ma)

print("_"*150)
MLPRegressorModel = MLPRegressor(activation='tanh',
                                 solver='lbfgs', 
                                 learning_rate='constant',
                                 alpha=0.00001 ,hidden_layer_sizes=(200, 100, 150),random_state=33)


MLPRegressorModel.fit(X_train, y_train)

print("_"*100)
print(MLPRegressorModel.score(X_train, y_train))
print(MLPRegressorModel.score(X_test, y_test))



KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 3, weights='uniform')
KNeighborsRegressorModel.fit(X_train, y_train)

KNeighborsRegressorModel.fit(X_train, y_train)

print("_"*100)
print(KNeighborsRegressorModel.score(X_train, y_train))
print(KNeighborsRegressorModel.score(X_test, y_test))



print("_"*150)
for x in range(2,20):
    Dt = DecisionTreeRegressor(max_depth=x,random_state=33)
    Dt.fit(X_train, y_train)

    print("x = ", x)
    print(Dt.score(X_train, y_train))
    print(Dt.score(X_test, y_test))
    print("_"*100)


autput = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# autput.to_csv("autput.csv",index=False)