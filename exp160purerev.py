import numpy as np

import sine_data as dataset
import copy
import random 
import intervene_noise_with_signal_reversed_input160

np.random.seed(0)
random.seed(0)
##########

list_1=np.random.randint(0,1600, size=133)
# print(list_1) 
list_2=np.random.randint(1600,3200, size=133)
# print(list_2) 
list_3=np.random.randint(3200,4800, size=133)
# print(list_3)
total_list=np.concatenate((list_1,list_2,list_3))
# print(total_list)

##########
data_pure_o, data_composition_o = dataset.main()
data_pure_lis=data_pure_o[total_list,:]
loss_matrix_comp_25rev = intervene_noise_with_signal_reversed_input160.main(data_pure_lis)
np.save('/home/rahul/ICML_IntGPT/intervention_noise/result/pure160rev', loss_matrix_comp_25rev)
