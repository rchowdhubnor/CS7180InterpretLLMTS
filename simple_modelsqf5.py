import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

np.random.seed(0)
random.seed(0)
randomList=[]
# traversing the loop 15 times
for i in range(200):
   # generating a random number in the range 1 to 100
   r=random.randint(0,1599)
   # checking whether the generated random number is not in the
   # randomList
   if r not in randomList:
      # appending the random number to the resultant list, if the condition is true
      randomList.append(r)

total_index = np.linspace(0,1599,1600)
total_index = total_index.astype(int)
train_index = np.delete(total_index, randomList[0:160])
test_index = randomList[0:160]
X_50_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_2f.npy')[train_index,:]
Y_50_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_2f.npy')[train_index,0:1]
X_50_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_sq5f.npy')[train_index,:]
Y_50_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_sq5f.npy')[train_index,0:1]

X_25_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_2f.npy')[train_index,:]
Y_25_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_2f.npy')[train_index,0:1]
X_25_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_sq5f.npy')[train_index,:]
Y_25_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_sq5f.npy')[train_index,0:1]
X_train=X_50_sq5f_train
Y_train=Y_50_sq5f_train
# Y_train=np.concatenate((Y_train,Y_25_sq5f_train),axis=0)
X_50_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_2f.npy')[test_index,:]
Y_50_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_2f.npy')[test_index,0:1]
X_50_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_sq5f.npy')[test_index,:]
Y_50_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_sq5f.npy')[test_index,0:1]

X_25_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_2f.npy')[test_index,:]
Y_25_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_2f.npy')[test_index,0:1]
X_25_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_sq5f.npy')[test_index,:]
Y_25_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_sq5f.npy')[test_index,0:1]
X_test=X_50_sq5f_test
Y_test=Y_50_sq5f_test


# print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
# print("polynomial of order 1")
# model = make_pipeline(PolynomialFeatures(1), LinearRegression())
# reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
# print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
# print("polynomial of order 2")
# model = make_pipeline(PolynomialFeatures(2), LinearRegression())
# reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
# print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
# print("polynomial of order 3")
# model = make_pipeline(PolynomialFeatures(3), LinearRegression())
# reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
# print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
mse_p=[]
x=np.linspace(1,10,10)
x=x.astype(int)
for i in range(46,47):
    for p in range(1,11):
        model = make_pipeline(PolynomialFeatures(p), LinearRegression())
        # print(X_train.shape,Y_train.shape)
        reg = model.fit((X_train[:,46:49]), Y_train[:,0:1])
        mse_p.append(mean_squared_error(Y_test[:,0:1], reg.predict((X_test[:,46:49]))))
plt.plot(x,mse_p)
plt.savefig('/home/rahul/ICML_IntGPT/find_model/plots_polyapp/mse_polysqf5.png')
estimators=[('poly', PolynomialFeatures(1)), ('linreg', LinearRegression())]
pipe = Pipeline(estimators)
final_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
regression = final_model.fit((X_train[:,46:49]), Y_train[:,0:1])
print(regression.steps[1][1].coef_,regression.steps[1][1].intercept_)
print(mean_squared_error(Y_test[:,0:1], regression.predict((X_test[:,46:49]))))