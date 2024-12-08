import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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
X_50_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_2f.npy')[train_index,25:]
Y_50_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_2f.npy')[train_index,0:1]
X_50_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_sq5f.npy')[train_index,25:]
Y_50_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_sq5f.npy')[train_index,0:1]

X_25_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_2f.npy')[train_index,:]
Y_25_2f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_2f.npy')[train_index,0:1]
X_25_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_sq5f.npy')[train_index,:]
Y_25_sq5f_train=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_sq5f.npy')[train_index,0:1]
X_train=np.concatenate((X_50_2f_train,X_50_sq5f_train,X_25_2f_train),axis=0)
X_train=np.concatenate((X_train,X_25_sq5f_train),axis=0)
Y_train=np.concatenate((Y_50_2f_train,Y_50_sq5f_train,Y_25_2f_train),axis=0)
Y_train=np.concatenate((Y_train,Y_25_sq5f_train),axis=0)
X_50_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_2f.npy')[test_index,25:]
Y_50_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_2f.npy')[test_index,0:1]
X_50_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_sq5f.npy')[test_index,25:]
Y_50_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_sq5f.npy')[test_index,0:1]

X_25_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_2f.npy')[test_index,:]
Y_25_2f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_2f.npy')[test_index,0:1]
X_25_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_25_sq5f.npy')[test_index,:]
Y_25_sq5f_test=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_25_sq5f.npy')[test_index,0:1]
X_test=np.concatenate((X_50_2f_test,X_50_sq5f_test,X_25_2f_test),axis=0)
X_test=np.concatenate((X_test,X_25_sq5f_test),axis=0)
Y_test=np.concatenate((Y_50_2f_test,Y_50_sq5f_test,Y_25_2f_test),axis=0)
Y_test=np.concatenate((Y_test,Y_25_sq5f_test),axis=0)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
print("polynomial of order 1")
model = make_pipeline(PolynomialFeatures(1), LinearRegression())
reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
print("polynomial of order 2")
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
print("polynomial of order 3")
model = make_pipeline(PolynomialFeatures(3), LinearRegression())
reg = model.fit(X_train[:,21:(21+3)], Y_train[:,0:1])
print("mse of three tokens",mean_squared_error(Y_test[:,0:1], reg.predict(X_test[:,21:(21+3)])))
mse_p=[]
x=np.linspace(1,20,20)
for i in range(20,21):
    for p in range(1,21):
        model = make_pipeline(PolynomialFeatures(p), LinearRegression())
        reg = model.fit((X_train[:,i:24]), Y_train[:,0:1])
        mse_p.append(mean_squared_error(Y_test[:,0:1], reg.predict((X_test[:,i:24]))))
plt.plot(x,mse_p)
plt.savefig('/home/rahul/ICML_IntGPT/find_model/plots_polyapp/mse_poly.png')

