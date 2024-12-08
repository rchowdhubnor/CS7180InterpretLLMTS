import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
original_train_data = np.load('/home/rahul/LLM_Research/Probing/data/activation/train/activation.npy')
original_test_data = np.load('/home/rahul/LLM_Research/Probing/data/activation/test/activation.npy')
print("original train data", original_train_data.shape)
print("original test data", original_test_data.shape)


list_train= np.linspace(768*16-1 ,768*32-1,768*16)
list_train = list_train.astype(int)
list_up_train = np.concatenate((list_train[0:768*4], list_train[768*4*3:]))
list_up_train = list_up_train.astype(int)
list_down_train = np.concatenate((list_train[768*4:768*8], list_train[768*8:768*12]))
list_down_train = list_down_train.astype(int)

train_data = np.concatenate((original_train_data[list_up_train],original_train_data[list_down_train]))

# left_index_train=768*8
# # left_index_test=77*8
# right_index=768*16
# # list_ = np.linspace(768*16 - 1,768*32,768*16)
# train_data = original_train_data[left_index_train:right_index]
# train_label = np.zeros((train_data.shape[0] ))
# test_data = original_test_data

logregvec=np.load('/home/rahul/LLM_Research/Probing/Classify/expupdown/logreg.npy')
root= "/home/rahul/LLM_Research/Probing/ProjectionPlots/expupdown"+"/"
comp1 = "Exponential Up"
comp2 = "Exponential Down" 
alpha1=0.005
alpha2=0.1
index_all= [np.random.randint(0, int(train_data.shape[0]//2)) for p in range(int(train_data.shape[0]//2))]
index_100 = [np.random.randint(0, int(100)) for p in range(int(100))]
index_list = [index_all,index_100]
for index in index_list:
    size_index = len(index)
    train_label = np.zeros((train_data.shape[0] ))
    train_label[0:int(train_data.shape[0]//2)]=0*np.ones((int(train_data.shape[0]//2)))
    train_label[int(train_data.shape[0]//2):]=1*np.ones((int(train_data.shape[0]//2)))
    train_label = train_label.astype(int)
    # test_label = np.zeros((test_data.shape[0] ))
    # test_label[0:int(test_data.shape[0]//2)]=0*np.ones((int(test_data.shape[0]//2)))
    # test_label[int(test_data.shape[0]//2):]=1*np.ones((int(test_data.shape[0]//2)))
    # test_label = test_label.astype(int)

    train = np.zeros((train_data.shape[0], train_data.shape[1], (train_data.shape[2]+1) ))
    train[:,:,:-1]=train_data
    # test = np.zeros((test_data.shape[0], test_data.shape[1], (test_data.shape[2]+1) ))
    # test[:,:,:-1]=test_data
    for i in range(train.shape[0]):
        for j in range(12):
            train[i,j,-1] = train_label[i]
    # for i in range(test.shape[0]):
    #     for j in range(12):
    #         test[i,j,-1] = test_label[i]
    ###### label 0 is exponential label 1 is periodic
    decider_pure = np.average(train[0:int(train.shape[0]//2),:,0:768], axis=0)
    decider_mix = np.average(train[int(train.shape[0]//2):,:,0:768], axis=0)

    diff_pure_mix =  decider_pure - decider_mix
    proj_train_euc_pure=np.zeros((int(train_data.shape[0]//2), 12))
    proj_train_euc_mix=np.zeros((int(train_data.shape[0]//2), 12))

    proj_train_euc_pure_pca=np.zeros((int(train_data.shape[0]//2), 12))
    proj_train_euc_mix_pca=np.zeros((int(train_data.shape[0]//2), 12))
    pcacomp=np.zeros((12,768))
    per_euc=np.zeros((12,768))

    for l in range(12):
            pca=PCA(1)
            pca.fit(train_data[:,l,:])
            pcacomp[l,:]=pca.components_[0]
            
    for n in range(int(train_data.shape[0])):
        for l in range(12):
            if n<int(train_data.shape[0]//2):
                proj_train_euc_pure[n,l]=(np.dot(train[n,l,:-1], diff_pure_mix[l,:]) / np.linalg.norm(diff_pure_mix[l,:]) )
                pca_proj=pcacomp[l,:]-(np.dot(pcacomp[l,:], diff_pure_mix[l,:]) / np.linalg.norm(diff_pure_mix[l,:]) )*diff_pure_mix[l,:]
                proj_train_euc_pure_pca[n,l]=(np.dot(train[n,l,:-1], pca_proj) / np.linalg.norm(pca_proj) )

                
            else:
                
                proj_train_euc_mix[n-train_data.shape[0]//2,l]=(np.dot(train[n,l,:-1], diff_pure_mix[l,:]) / np.linalg.norm(diff_pure_mix[l,:]) )
                pca_proj=pcacomp[l,:]-(np.dot(pcacomp[l,:], diff_pure_mix[l,:]) / np.linalg.norm(diff_pure_mix[l,:]) )*diff_pure_mix[l,:]
                proj_train_euc_mix_pca[n-train_data.shape[0]//2,l]=(np.dot(train[n,l,:-1], pca_proj) / np.linalg.norm(pca_proj) )

                


    fig, axs = plt.subplots(4, 3,figsize=(12,16))
    count=0
    for i in range(4):
        for j in range(3):
            axs[i,j].scatter(proj_train_euc_pure[:,count], proj_train_euc_pure_pca[:,count],c='red', alpha=alpha1)
            axs[i,j].scatter(proj_train_euc_mix[:,count],proj_train_euc_mix_pca[:,count] ,c='yellow',alpha=alpha2)
            axs[i,j].set_title("Layer "+str(count+1))
            axs[i,j].set_ylabel("$\perp$ PCA 1")
            axs[i,j].set_xlabel("Projection on Difference of Centroids Vec")
            axs[i,j].legend([comp1 + "(Red)", comp2 + "(Yellow)"], loc="upper right")
            axs[i,j].grid(False)
            # axs[i,j].axis('off')
            axs[i,j].set_yticklabels([])
            axs[i,j].set_xticklabels([])
            count+=1

    plt.show()
    plt.savefig(root+"pca_euc"+str(size_index)+".png")
    plt.clf()
    fig, axs = plt.subplots(4, 3,figsize=(12,16))
    count=0
    dummy=np.zeros_like(proj_train_euc_pure)
    for i in range(4):
        for j in range(3):
            axs[i,j].scatter(proj_train_euc_pure[:,count], dummy[:,count],c='red', alpha=alpha1)
            axs[i,j].scatter(proj_train_euc_mix[:,count],dummy[:,count],c='yellow',alpha=alpha2)
            axs[i,j].set_title("Layer "+str(count+1))
            axs[i,j].set_xlabel("Projection on Difference of Centroids Vec")
            axs[i,j].set_ylabel("")
            axs[i,j].legend([comp1+ "(Red)", comp2+ "(Yellow)"], loc="upper right")
            axs[i,j].grid(False)
            # axs[i,j].axis('off')
            axs[i,j].set_yticklabels([])
            axs[i,j].set_xticklabels([])
            count+=1

    plt.show()
    plt.savefig(root+"euc1D"+str(size_index)+".png")
    plt.clf()

    proj_train_log_pure=np.zeros((int(train_data.shape[0]//2), 12))
    proj_train_log_mix=np.zeros((int(train_data.shape[0]//2), 12))
    proj_train_logpca_pure=np.zeros((int(train_data.shape[0]//2), 12))
    proj_train_logpca_mix=np.zeros((int(train_data.shape[0]//2), 12))
    for n in range(int(train_data.shape[0])):
        for l in range(12):
            if n<int(train_data.shape[0]//2):
                proj_train_log_pure[n,l]=(np.dot(train[n,l,:-1], logregvec[l,:]) / np.linalg.norm(logregvec[l,:]) )
                log=pcacomp[l,:]-(np.dot(pcacomp[l,:], logregvec[l,:]) / np.linalg.norm(logregvec[l,:]) )*logregvec[l,:]
                proj_train_logpca_pure[n,l]=(np.dot(train[n,l,:-1], log) / np.linalg.norm(log) )
            else:
                proj_train_log_mix[n-train_data.shape[0]//2,l]=(np.dot(train[n,l,:-1], logregvec[l,:]) / np.linalg.norm(logregvec[l,:]) )
                log=pcacomp[l,:]-(np.dot(pcacomp[l,:], logregvec[l,:]) / np.linalg.norm(logregvec[l,:]) )*logregvec[l,:]
                proj_train_logpca_mix[n-train_data.shape[0]//2,l]=(np.dot(train[n,l,:-1], log) / np.linalg.norm(log) )
                
    ###################################
    fig, axs = plt.subplots(4, 3,figsize=(12,16))
    count=0
    
    for i in range(4):
        for j in range(3):

            axs[i,j].scatter(proj_train_log_pure[:,count], proj_train_logpca_pure[:,count],c='red', alpha=alpha1)
            axs[i,j].scatter(proj_train_log_mix[:,count],  proj_train_logpca_mix[:,count] ,c='yellow',alpha=alpha2)
            axs[i,j].set_title("Layer "+str(count+1))
            axs[i,j].set_ylabel("$\perp$ PCA 1")
            axs[i,j].set_xlabel("Projection on Weight Vector ")
            axs[i,j].legend([comp1+"(Red)", comp2+"(Yellow)"], loc="upper right")
            axs[i,j].grid(False)
            # axs[i,j].axis('off')
            axs[i,j].set_yticklabels([])
            axs[i,j].set_xticklabels([])
            count+=1

    plt.show()
    plt.savefig(root+"weight_pca"+str(size_index) +".png")
    plt.clf()
    fig, axs = plt.subplots(4, 3,figsize=(12,16))
    count=0
    dummy=np.zeros_like(proj_train_euc_pure)
    for i in range(4):
        for j in range(3):
            axs[i,j].scatter(proj_train_log_pure[:,count], dummy[:,count],c='red', alpha=alpha1)
            axs[i,j].scatter(proj_train_log_mix[:,count],dummy[:,count],c='yellow',alpha=alpha2)
            axs[i,j].set_title("Layer "+str(count+1))
            axs[i,j].set_xlabel("Projection on Weight Vector")
            axs[i,j].set_ylabel("")
            axs[i,j].legend([comp1+"(Red)", comp2+"(Yellow)"], loc="upper right")
            axs[i,j].grid(False)
            # axs[i,j].axis('off')
            axs[i,j].set_yticklabels([])
            axs[i,j].set_xticklabels([])
            count+=1

    plt.show()
    plt.savefig(root+"weight1D"+str(size_index)+".png")
    plt.clf()
            
            