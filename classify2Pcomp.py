import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random 
random.seed(0)

np.random.seed(0)
############classifier 2f

###########################

data2f40 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose24f80.npy')
data4f20 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose28f80.npy')
data8f10 = np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose48f40.npy')

label2f40 = np.ones((1600))
label4f20 = np.zeros((1600))
label8f10 = np.zeros((1600))

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


train_data = np.concatenate([data2f40[res,:,:],data4f20[res,:,:],data8f10[res,:,:]])
test_data = np.concatenate([data2f40[randomList[0:100],:,:],data4f20[randomList[0:100],:,:],data8f10[randomList[0:100],:,:]])

train_label = np.concatenate([label2f40[res],label4f20[res],label8f10[res]])
test_label = np.concatenate([label2f40[randomList[0:100]],label4f20[randomList[0:100]],label8f10[randomList[0:100]]])

print("train data", train_data.shape)
print("test data", test_data.shape)
# print(test_label)


# # np.random.shuffle(train)
c=0.1
logistic_training_loss2f=[]
logistic_testing_loss2f=[]
lin_class2f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class2f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label.astype(int)
    lin_class2f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class2f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class2f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    logistic_training_loss2f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label.astype(int)
    testing_score =  lin_class2f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    logistic_testing_loss2f.append(testing_score)
     
print("logistic training accuracy 24f:\n", logistic_training_loss2f)
print("logistic testing accuracy 24f:\n", logistic_testing_loss2f)


######################################## 4f


label2f40_4 = np.zeros((1600))
label4f20_4 = np.ones((1600))
label8f10_4 = np.zeros((1600))



train_label_4 = np.concatenate([label2f40_4[res],label4f20_4[res],label8f10_4[res]])
test_label_4 = np.concatenate([label2f40_4[randomList[0:100]],label4f20_4[randomList[0:100]],label8f10_4[randomList[0:100]]])

# print(test_label)


# # np.random.shuffle(train)
c=0.1
logistic_training_loss4f=[]
logistic_testing_loss4f=[]
lin_class4f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class4f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label_4.astype(int)
    lin_class4f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class4f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class4f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    logistic_training_loss4f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label_4.astype(int)
    testing_score =  lin_class4f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    logistic_testing_loss4f.append(testing_score)
     
print("logistic training accuracy 28f:\n", logistic_training_loss4f)
print("logistic testing accuracy 28f:\n", logistic_testing_loss4f)
######################################## 4f
######################################## 8f
label2f40_8 = np.zeros((1600))
label4f20_8 = np.zeros((1600))
label8f10_8 = np.ones((1600))



train_label_8 = np.concatenate([label2f40_8[res],label4f20_8[res],label8f10_8[res]])
test_label_8 = np.concatenate([label2f40_8[randomList[0:100]],label4f20_8[randomList[0:100]],label8f10_8[randomList[0:100]]])

# print(test_label)


# # np.random.shuffle(train)
c=0.1
logistic_training_loss8f=[]
logistic_testing_loss8f=[]
lin_class8f = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class8f['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=10000)
    # lab_train=train_data[:,i,:]
    lab_train= train_label_8.astype(int)
    lin_class8f['lr_clf_layer' + str(i)].fit(train_data[:,i,:], lab_train)
    logreg_comp[i,:]=lin_class8f['lr_clf_layer' + str(i)].coef_
    training_score = lin_class8f['lr_clf_layer' + str(i)].score(train_data[:,i,:], lab_train)
    logistic_training_loss8f.append(training_score)
    # lab_test=test_data[:,i]
    lab_test= test_label_8.astype(int)
    testing_score =  lin_class8f['lr_clf_layer' + str(i)].score(test_data[:,i,:], lab_test)
    logistic_testing_loss8f.append(testing_score)
     
print("logistic training accuracy 48f:\n", logistic_training_loss8f)
print("logistic testing accuracy 48f:\n", logistic_testing_loss8f)
######################################## 8f
#####################Test Compose Zero Shot
# test_data_compose28_120=np.load('/home/rahul/ICML_IntGPT/probing/data/activation/compose48f20.npy')

# lab_test_28_120_2=np.ones(test_data_compose28_120.shape[0])

# logistic_testing_loss28_120_2=[]

# for i in range(12):

#     lab_test= lab_test_28_120_2.astype(int)
#     testing_score =  lin_class8f['lr_clf_layer' + str(i)].score(test_data_compose28_120[:,i,:], lab_test)
#     logistic_testing_loss28_120_2.append(testing_score)
     

# print("logistic testing accuracy 24_40_2:\n", logistic_testing_loss28_120_2)

# ############################# Test Compose 8f
# lab_test_24_120_8=np.ones(test_data_compose28_120.shape[0])

# logistic_testing_loss28_120_8=[]

# for i in range(12):

#     lab_test= lab_test_24_120_8.astype(int)
#     testing_score =  lin_class4f['lr_clf_layer' + str(i)].score(test_data_compose28_120[:,i,:], lab_test)
#     logistic_testing_loss28_120_8.append(testing_score)
     

# print("logistic testing accuracy 28_120_4:\n", logistic_testing_loss28_120_8)


#############################
