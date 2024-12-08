import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random 
random.seed(0)

np.random.seed(0)
############classifier 2f

###########################

data2f40 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/pure2f40.npy')
data4f20 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/pure4f20.npy')
data8f10 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/pure8f10.npy')
data24f = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose24f40.npy')
data28f = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose28f40.npy')
data48f = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose48f20.npy')


data2f40,data4f20,
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

############################## 2f
train_data2f = np.concatenate([data2f40[res,:,:],data24f[res,:,:],data28f[res,:,:],data4f20[res,:,:],data8f10[res,:,:],data48f[res,:,:]])
test_data2f = np.concatenate([data2f40[randomList[0:100]],data24f[randomList[0:100]],data28f[randomList[0:100]],data4f20[randomList[0:100]],data8f10[randomList[0:100]],data48f[randomList[0:100]]])
# test_data2f = np.concatenate([data24f[randomList[0:100]],data28f[randomList[0:100]],data48f[randomList[0:100]]])


train_labelpos2f = np.ones(train_data2f.shape[0]//2)
train_labelneg2f = np.zeros(train_data2f.shape[0]//2)

train_label2f = np.concatenate([train_labelpos2f,train_labelneg2f])

test_labelpos2f = np.ones(test_data2f.shape[0]//2)
test_labelneg2f = np.zeros(test_data2f.shape[0]//2)
# test_labelpos2f = np.ones(200)
# test_labelneg2f = np.zeros(100)
test_label2f = np.concatenate([test_labelpos2f,test_labelneg2f])




# # np.random.shuffle(train)
c=0.1
logistic_training_loss2f=[]
logistic_testing_loss2f=[]
lin_class2f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class2f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label2f.astype(int)
    lin_class2f['lr_clf_layer' + str(i)].fit(train_data2f[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class2f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class2f['lr_clf_layer' + str(i)].score(train_data2f[:,i,:], lab_train)
    logistic_training_loss2f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label2f.astype(int)
    testing_score =  lin_class2f['lr_clf_layer' + str(i)].score(test_data2f[:,i,:], lab_test)
    logistic_testing_loss2f.append(testing_score)
     
print("logistic training accuracy 2f:\n", logistic_training_loss2f)
print("logistic testing accuracy 2f:\n", logistic_testing_loss2f)
################################### 2f 


################################### 4f

train_data4f = np.concatenate([data24f[res,:,:],data4f20[res,:,:],data48f[res,:,:],data28f[res,:,:],data8f10[res,:,:],data2f40[res,:,:]])
test_data4f = np.concatenate([data24f[randomList[0:100]],data4f20[randomList[0:100]],data48f[randomList[0:100]],data28f[randomList[0:100]],data8f10[randomList[0:100]],data2f40[randomList[0:100]]])
# test_data4f = np.concatenate([data24f[randomList[0:100]],data48f[randomList[0:100]],data28f[randomList[0:100]]])


train_labelpos4f = np.ones(train_data4f.shape[0]//2)
train_labelneg4f = np.zeros(train_data4f.shape[0]//2)
train_label4f = np.concatenate([train_labelpos4f,train_labelneg4f])

test_labelpos4f = np.ones(test_data4f.shape[0]//2)
test_labelneg4f = np.zeros(test_data4f.shape[0]//2)
# test_labelpos4f = np.ones(200)
# test_labelneg4f = np.zeros(100)
test_label4f = np.concatenate([test_labelpos4f,test_labelneg4f])




# # np.random.shuffle(train)
c=0.1
logistic_training_loss4f=[]
logistic_testing_loss4f=[]
lin_class4f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class4f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label4f.astype(int)
    lin_class4f['lr_clf_layer' + str(i)].fit(train_data4f[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class4f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class4f['lr_clf_layer' + str(i)].score(train_data4f[:,i,:], lab_train)
    logistic_training_loss4f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label4f.astype(int)
    testing_score =  lin_class4f['lr_clf_layer' + str(i)].score(test_data4f[:,i,:], lab_test)
    logistic_testing_loss4f.append(testing_score)
     
print("logistic training accuracy 4f:\n", logistic_training_loss4f)
print("logistic testing accuracy 4f:\n", logistic_testing_loss4f)


################################### 4f

################################### 8f

train_data8f = np.concatenate([data48f[res,:,:],data28f[res,:,:],data8f10[res,:,:], data24f[res,:,:],data4f20[res,:,:],data2f40[res,:,:]])
test_data8f = np.concatenate([data48f[randomList[0:100]],data28f[randomList[0:100]],data8f10[randomList[0:100]], data24f[randomList[0:100]],data4f20[randomList[0:100]],data2f40[randomList[0:100]]])
# test_data8f = np.concatenate([data48f[randomList[0:100]],data28f[randomList[0:100]], data24f[randomList[0:100]]])

train_labelpos8f = np.ones(train_data8f.shape[0]//2)
train_labelneg8f = np.zeros(train_data8f.shape[0]//2)
train_label8f = np.concatenate([train_labelpos8f,train_labelneg8f])

test_labelpos8f = np.ones(test_data8f.shape[0]//2)
test_labelneg8f = np.zeros(test_data8f.shape[0]//2)
# test_labelpos8f = np.ones(200)
# test_labelneg8f = np.zeros(100)
test_label8f = np.concatenate([test_labelpos8f,test_labelneg8f])




# # np.random.shuffle(train)
c=0.1
logistic_training_loss8f=[]
logistic_testing_loss8f=[]
lin_class8f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class8f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label8f.astype(int)
    lin_class8f['lr_clf_layer' + str(i)].fit(train_data8f[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class8f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class8f['lr_clf_layer' + str(i)].score(train_data8f[:,i,:], lab_train)
    logistic_training_loss8f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label8f.astype(int)
    testing_score =  lin_class8f['lr_clf_layer' + str(i)].score(test_data8f[:,i,:], lab_test)
    logistic_testing_loss8f.append(testing_score)
     
print("logistic training accuracy 8f:\n", logistic_training_loss8f)
print("logistic testing accuracy 8f:\n", logistic_testing_loss8f)
################################### 8f