import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
###########################

original_train_data = np.load('/home/rahul/LLM_Research/Probing/data/activation/train/activation.npy')
original_test_data = np.load('/home/rahul/LLM_Research/Probing/data/activation/test/activation.npy')



# ######################
list_train= np.linspace(768*16-1 ,768*32-1,768*16)
list_train = list_train.astype(int)
list_up_train = np.concatenate((list_train[0:768*4], list_train[768*4*3:]))
list_up_train = list_up_train.astype(int)
list_down_train = np.concatenate((list_train[768*4:768*8], list_train[768*8:768*12]))
list_down_train = list_down_train.astype(int)

train_data = np.concatenate((original_train_data[list_up_train],original_train_data[list_down_train]))
######################
list_test= np.linspace(77*16-1 ,77*32-1,77*16)
list_test = list_test.astype(int)
list_up_test = np.concatenate((list_test[0:77*4], list_test[77*4*3:]))
list_up_test = list_up_test.astype(int)
list_down_test = np.concatenate((list_test[77*4:77*8], list_test[77*8:77*12]))
list_down_test = list_down_test.astype(int)

test_data = np.concatenate((original_test_data[list_up_test],original_test_data[list_down_test]))







######################################################################

left_index_train=768*8
left_index_test=77*8
right_index_train=768*16
right_index_test=77*16
# right_index=768*16
# list_ = np.linspace(768*16 - 1,768*32,768*16)
# train_data = original_train_data [left_index_train:right_index_train]
train_label = np.zeros((train_data.shape[0] ))
# test_data = original_test_data [left_index_test:right_index_test]
test_label = np.zeros((test_data.shape[0] ))
print(train_data.shape[0]//2)
print(test_data.shape[0]//2)
right_index_train=int(train_data.shape[0]//2 )
right_index_test=int(test_data.shape[0]//2 )
train_label[0:int(train_data.shape[0]//2)]=1*np.ones((int(right_index_train)))
train_label[right_index_train:]=2*np.ones((int(train_data.shape[0]-right_index_train)))
train_label = train_label.astype(int)
test_label = np.zeros((test_data.shape[0] ))
test_label[0:int(right_index_test)]=1*np.ones((int(right_index_test)))
test_label[int(right_index_test):]=2*np.ones((int(test_data.shape[0]-right_index_test)))
test_label = test_label.astype(int)


train = np.zeros((train_data.shape[0], train_data.shape[1], (train_data.shape[2]+1) ))
train[:,:,:-1]=train_data
test = np.zeros((test_data.shape[0], test_data.shape[1], (test_data.shape[2]+1) ))
test[:,:,:-1]=test_data
for i in range(train.shape[0]):
    for j in range(12):
        train[i,j,-1] = train_label[i]
for i in range(test.shape[0]):
    for j in range(12):
        test[i,j,-1] = test_label[i]
###### label 0 is exponential label 1 is periodic
decider_exp = np.average(train[0:int(train.shape[0]//2),:,0:768], axis=0)
decider_sine = np.average(train[int(train.shape[0]//2):,:,0:768], axis=0)
train_pred = np.zeros((train.shape[0], 12))
test_pred =  np.zeros((test.shape[0], 12))
train_acc_count = np.zeros((1,12))
test_acc_count = np.zeros((1,12))
training_accuracy = np.zeros((1,12))
testing_accuracy = np.zeros((1,12))
for i in range(train.shape[0]):
    for j in range(12):
        if np.linalg.norm(train[i,j,:-1]-decider_exp[j,:]) >=  np.linalg.norm(train[i,j,:-1]-decider_sine[j,:]):
            train_pred[i,j]=2

        else:
            train_pred[i,j]=1
        if train_pred[i,j]==train[i,j,-1]:
            train_acc_count[0,j] +=1

for j in range(12):
    training_accuracy[0,j]=(train_acc_count[0,j]/train.shape[0])

for i in range(test.shape[0]):
    for j in range(12):
        if np.linalg.norm(test[i,j,:-1]-decider_exp[j,:]) >=  np.linalg.norm(test[i,j,:-1]-decider_sine[j,:]):
            test_pred[i,j]=2
        else:
            test_pred[i,j]=1
        if test_pred[i,j]==test[i,j,-1]:
            test_acc_count[0,j] +=1
        # print(test[i,j,-1])
for j in range(12):
    testing_accuracy[0,j]=(test_acc_count[0,j]/test.shape[0])
print("euclidean training accuracy\n", training_accuracy)
print("euclidean testing accuracy:\n", testing_accuracy)

# print("done")


            

# np.random.shuffle(train)
c=0.1
logistic_training_loss=[]
logistic_testing_loss=[]
lin_class = {}
logreg_comp=np.zeros((12,768))
for i in range(12):
    lin_class['lr_clf_layer' + str(i)] = LogisticRegression(penalty='l2',C=c,max_iter=1500)
    lab_train=train[:,i,-1]
    lab_train= lab_train.astype(int)
    lin_class['lr_clf_layer' + str(i)].fit(train[:,i,:-1], lab_train)
    logreg_comp[i,:]=lin_class['lr_clf_layer' + str(i)].coef_
    training_score = lin_class['lr_clf_layer' + str(i)].score(train[:,i,:-1], lab_train)
    logistic_training_loss.append(training_score)
    lab_test=test[:,i,-1]
    lab_test= lab_test.astype(int)
    testing_score =  lin_class['lr_clf_layer' + str(i)].score(test[:,i,:-1], lab_test)
    logistic_testing_loss.append(testing_score)

                                                           
print("logistic training accuracy:\n", logistic_training_loss)
print("logistic testing accuracy:\n", logistic_testing_loss)
np.save('/home/rahul/LLM_Research/Probing/Classify/expupdown/logreg', logreg_comp)
print("Thank You")