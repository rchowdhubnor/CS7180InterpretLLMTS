from nnsight import LanguageModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import sine_data as dataset
import copy
llm = LanguageModel("openai-community/gpt2", device_map="auto")
data_pure_o, data_composition_o = dataset.main()
# print(data_pure_o.shape)
# print(data_composition_o.shape)
Row_len=[(0,1600),(1600,3200),(3200,4800)]
Pure_Set=[(40,80,120),(20,40,60),(10,20,30)]
Composition_Set=[(40,80,120),(40,80,120),(20,40,60)]


Dataset=[data_pure_o, data_composition_o]
Len_Set=[Pure_Set,Composition_Set]
var_name=["pure","compose"]
freq_name=[["2f","4f","8f"],["24f","28f","48f"]]
count=0
for d in range(2):
    original_data=copy.deepcopy(Dataset[d])
    for r in range(3):
        l_row,r_row=Row_len[r]
        for c in range(3):
            data_=copy.deepcopy(original_data[l_row:r_row,0:Len_Set[d][r][c]])
            data = data_.astype(int)
            # print(data.shape)
            min_max=np.zeros((data.shape[0], 2))
            count+=1
            
            
            for n in range(data.shape[0]):
                td_ = copy.deepcopy(data[n,:])
                td=np.array([int(i) for i in copy.deepcopy(td_)]) 
                min_max[n,0]=np.min(td)
                min_max[n,1]=np.max(td)
                

            np.save('/home/rahul/ICML_IntGPT/probing_regression/min_max/'+var_name[d]+str(freq_name[d][r])+str(Len_Set[d][r][c]),min_max)

print(count)
