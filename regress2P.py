import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import random 
# import sine_data as dataset
import copy


###############################################################################


###############################################################################



random.seed(0)

np.random.seed(0)
############classifier 2f

###########################

data2f40 = np.load('/home/rahul/ICML_IntGPT/probing_regression/activation/pure2f80.npy')
data4f20 = np.load('/home/rahul/ICML_IntGPT/probing_regression/activation/pure4f40.npy')
data8f10 = np.load('/home/rahul/ICML_IntGPT/probing_regression/activation/pure8f20.npy')

label2f40 = np.load('/home/rahul/ICML_IntGPT/probing_regression/min_max/pure2f80.npy')
label4f20 = np.load('/home/rahul/ICML_IntGPT/probing_regression/min_max/pure4f40.npy')
label8f10 = np.load('/home/rahul/ICML_IntGPT/probing_regression/min_max/pure8f20.npy')

randomList=[]
# traversing the loop 15 times
for i in range(150):
   # generating a random number in the range 1 to 100
   r=random.randint(0,1599)
   # checking whether the generated random number is not in the
   # randomList
   if r not in randomList:
      # appending the random number to the resultant list, if the condition is true
      randomList.append(r)

total_index = np.linspace(0,1599,1600)
total_index = total_index.astype(int)
res = np.delete(total_index, randomList[0:100])


############################################ 2f
c=0.1
train_data=data2f40[res,:,:]
test_data=data2f40[randomList[0:100],:,:]

train_label=label2f40[res,:]
test_label=label2f40[randomList[0:100],:]
linear_training_loss2f=[]
linear_testing_loss2f=[]
lin_reg2f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_reg2f['lr_clf_layer' + str(i)] = LinearRegression()
    
    lab_train= train_label[:,1].astype(int)
    lin_reg2f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    # logreg_comp[i,:]=lin_reg2f['lr_clf_layer' + str(i)].coef_
    training_score = lin_reg2f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    linear_training_loss2f.append(training_score)
    
    lab_test= test_label[:,1].astype(int)
    testing_score =  lin_reg2f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    linear_testing_loss2f.append(testing_score)
     
print("linear training accuracy 2f:\n", linear_training_loss2f)
print("linear testing accuracy 2f:\n", linear_testing_loss2f)


# ######################################## 4f
c=0.1
train_data=data4f20[res,:,:]
test_data=data4f20[randomList[0:100],:,:]

train_label=label4f20[res,:]
test_label=label4f20[randomList[0:100],:]
linear_training_loss4f20=[]
linear_testing_loss4f20=[]
lin_reg4f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_reg4f['lr_clf_layer' + str(i)] = LinearRegression()
    
    lab_train= train_label[:,1].astype(int)
    lin_reg4f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    # logreg_comp[i,:]=lin_reg2f['lr_clf_layer' + str(i)].coef_
    training_score = lin_reg4f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    linear_training_loss4f20.append(training_score)
    
    lab_test= test_label[:,1].astype(int)
    testing_score =  lin_reg4f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    linear_testing_loss4f20.append(testing_score)
     
print("linear training accuracy 4f:\n", linear_training_loss4f20)
print("linear testing accuracy 4f:\n", linear_testing_loss4f20)

# ######################################## 4f
# ######################################## 8f
c=0.1
train_data=data8f10[res,:,:]
test_data=data8f10[randomList[0:100],:,:]

train_label=label8f10[res,:]
test_label=label8f10[randomList[0:100],:]
linear_training_loss8f=[]
linear_testing_loss8f=[]
lin_reg8f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_reg8f['lr_clf_layer' + str(i)] = LinearRegression()
    
    lab_train= train_label[:,1].astype(int)
    lin_reg8f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    # logreg_comp[i,:]=lin_reg2f['lr_clf_layer' + str(i)].coef_
    training_score = lin_reg8f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    linear_training_loss8f.append(training_score)
    
    lab_test= test_label[:,1].astype(int)
    testing_score =  lin_reg8f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    linear_testing_loss8f.append(testing_score)
     
print("linear training accuracy 8f:\n", linear_training_loss8f)
print("linear testing accuracy 8f:\n", linear_testing_loss8f)
# ######################################## 8f

