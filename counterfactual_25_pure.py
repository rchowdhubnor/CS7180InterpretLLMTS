from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import sine_data as dataset
# import sine_data_irrational as dataset
import copy
import random 
import time
start_time = time.time()
def is_number(n):
    try:
            float(n)  
    except ValueError:
            return False
    return True

np.random.seed(0)
random.seed(0)
##########

list_1=np.random.randint(0,1600, size=133)
# print(list_1) 
# list_2=np.random.randint(1600,3200, size=133)
# # print(list_2) 
# list_3=np.random.randint(3200,4800, size=133)
# # print(list_3)
# total_list=np.concatenate((list_1,list_2,list_3))
# print(total_list)

##########
data_pure_o, data_composition_o = dataset.main()
data_pure_lis=data_pure_o[list_1,:]

llm = LanguageModel("openai-community/gpt2", device_map="auto")


count=0
Loss_Data=np.zeros((data_pure_lis.shape[0],2,49))
print("Loss Shape:",Loss_Data.shape)


data_=copy.deepcopy(data_pure_lis)
data_ = data_.astype(int)


count+=1
st_list=np.linspace(0,24,25)
st_list=st_list.astype(int)
for n in range(data_.shape[0]):
    #################
    
    deg_=0
    for deg in [1]:
        s=0
        for steps in st_list:
            
            pd_o = copy.deepcopy(data_[n,:])
            if steps<24:
                pd_o[steps]=pd_o[steps]+int(deg*random.randint(np.min(data_), np.max(data_)))
            pd=np.array([int(i) for i in copy.deepcopy(pd_o[0:50])]) 
            # print("shape of pd", pd)
            pd_p=copy.deepcopy(pd[:-1])
            # print("shape of pdp",pd_p)
            pds=', '.join(map(str, pd_p))
            pd_s = pds + str(",")
            with llm.trace(pd_s) as tracer:
                exp_token1ni = llm.lm_head.output.argmax(dim=-1).save()
            pd_s_sni_sl = pd_s + llm.tokenizer.decode(exp_token1ni[0][-1])
            with llm.trace(pd_s_sni_sl) as tracer:
                exp_token2ni = llm.lm_head.output.argmax(dim=-1).save()
            pd_sni_sl_f= pd_s_sni_sl + llm.tokenizer.decode(exp_token2ni[0][-1])
            with llm.trace(pd_sni_sl_f) as tracer:
                exp_token3ni = llm.lm_head.output.argmax(dim=-1).save()
            exp_output1ni=llm.tokenizer.decode(exp_token1ni[0][-1])
            exp_output1ni=exp_output1ni.replace(" ", "")
            exp_output2ni=llm.tokenizer.decode(exp_token2ni[0][-1])
            exp_output2ni=exp_output2ni.replace(" ", "")
            exp_output3ni=llm.tokenizer.decode(exp_token3ni[0][-1])
            exp_output3ni=exp_output3ni.replace(" ", "")
            if exp_output1ni == '-':
                if is_number(exp_output2ni) and is_number(exp_output3ni):
                        rni=int(exp_output1ni+exp_output2ni+exp_output3ni)
                        
                        Loss_Per_Datani=abs(pd[-1]-rni)
                        if steps<48:
                            Loss_Data[n,1,s]=abs(pd_o[steps]-rni)
                elif is_number(exp_output2ni) :
                        rni=int(exp_output1ni+exp_output2ni)
                        Loss_Per_Datani=abs(pd[-1]-rni)
                        if steps<48:
                            Loss_Data[n,1,s]=abs(pd_o[steps]-rni)
                else:
                        Loss_Per_Datani = abs(pd[-1])
                        if steps<48:
                            Loss_Data[n,1,s]=abs(pd_o[steps])
            elif is_number(exp_output1ni) and is_number(exp_output2ni) :#####
                rni=int(exp_output1ni+exp_output2ni)
                Loss_Per_Datani=abs(pd[-1]-rni)
                if steps<48:
                    Loss_Data[n,1,s]=abs(pd_o[steps]-rni)
            elif is_number(exp_output1ni):
                rni=int(exp_output1ni)
                Loss_Per_Datani=abs(pd[-1]-rni)
                if steps<48:
                    Loss_Data[n,1,s]=abs(pd_o[steps]-rni)
            else:
                Loss_Per_Datani = abs(pd[-1])
                if steps<48:
                    Loss_Data[n,1,s]=abs(pd_o[steps])
            Loss_Data[n,0,s]=Loss_Per_Datani
            
            s+=1
        deg_+=1


np.save('/home/rahul/ICML_IntGPT/token_ranking/Loss_Mat/Loss_25_pure',Loss_Data)
