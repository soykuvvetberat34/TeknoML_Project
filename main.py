from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datas=pd.read_csv("/content/Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","NewLeague","Division"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","NewLeague","Division"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=99)

RFR=RandomForestRegressor()
RFR_params={
    "n_estimators":[50,100,150],
    "max_depth":[10,20,30],
    "min_samples_split":[2,4],
    "max_features":[1,2,3]
}

RFR_cv=GridSearchCV(RFR,RFR_params,cv=2,n_jobs=-1,verbose=2)
RFR_cv.fit(x_train,y_train)
n_estimators=RFR_cv.best_params_["n_estimators"]
max_depth=RFR_cv.best_params_["max_depth"]
min_samples_split=RFR_cv.best_params_["min_samples_split"]
max_features=RFR_cv.best_params_["max_features"]
RFR_tuned=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,max_features=max_features)
RFR_tuned.fit(x_train,y_train)
predict=RFR_tuned.predict(x_test)
MSE=mean_squared_error(y_test,predict)
RMSE=np.sqrt(MSE)
MAE=mean_absolute_error(y_test,predict)
print("\n Metrik sonuçları\n")
print("MSE:",MSE)
print("RMSE:",RMSE)
print("MAE:",MAE)

print("\n------------------------------------------\n")
print("Bağımsız değişkenlerin önem sıralaması\n")
Importance=pd.DataFrame({"Importance":RFR_tuned.feature_importances_*100},
                        index=x_train.columns)
Importance.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color="r")
plt.xlabel("Variable Importance")
plt.gca().legend_=None
plt.show()

print("\nDivision - Salary Grafiği\n")
sns.lineplot(datas,x="Division",y="Salary")
plt.show()

print("\n\nHistogram grafiği\n")
datas.hist(figsize=(12,12));



