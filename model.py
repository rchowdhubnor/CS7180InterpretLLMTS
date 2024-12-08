import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random 
import matplotlib.pyplot as plt
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
X=np.load('/home/rahul/ICML_IntGPT/find_model/data/Data_50_2f.npy')
Y=np.load('/home/rahul/ICML_IntGPT/find_model/data/Loss_50_2f.npy')

# Loss_Matrix=[model prediction,ground truth]
train_data=X[train_index,:]
train_label=Y[train_index,:]
test_data=X[test_index,:]
test_label=Y[test_index,:]
print(train_data.shape,test_data.shape,train_label.shape,test_label.shape)
print(train_data[0,:])
print(train_label[0,:])
mse_mat=[]

print("last tokens model")
reg = LinearRegression().fit(train_data[:,46:(46+3)], train_label[:,0:1])
print("mse of three tokens",mean_squared_error(test_label[:,0], reg.predict(test_data[:,46:(46+3)])))
print("three tokens",reg.coef_,reg.intercept_)
reg = LinearRegression().fit(train_data[:,47:(47+2)], train_label[:,0:1])
print("mse of two tokens",mean_squared_error(test_label[:,0], reg.predict(test_data[:,47:(47+2)])))
print("three tokens",reg.coef_,reg.intercept_)
reg = LinearRegression().fit(train_data[:,48:(48+1)], train_label[:,0:1])
print("mse of one token",mean_squared_error(test_label[:,0], reg.predict(test_data[:,48:(48+1)])))
print("three tokens",reg.coef_,reg.intercept_)

