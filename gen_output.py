import sine_data as rational_data
import sine_data_irrational as irrational_data
import numpy as np
import copy
import time
import random
np.random.seed(0)
random.seed(0) 
from nnsight import LanguageModel
llm = LanguageModel("openai-community/gpt2", device_map="auto")

start_time = time.time()
def is_number(n):
    try:
            float(n)  
    except ValueError:
            return False
    return True

data_rat, _ = rational_data.main()
data_irr, _ = irrational_data.main()
data=data_irr[1600:1600*2,0:25]
# data=data_rat[0:1600,0:25]
# data=np.concatenate(data_r,data_i)
print(data.shape)
llm = LanguageModel("openai-community/gpt2", device_map="auto")
Data_Matrix=copy.deepcopy(data)
# Loss_Matrix=[model prediction,ground truth]
Loss_Matrix=np.zeros((data.shape[0],2))
for n in range(data.shape[0]):
      pd_o = copy.deepcopy(data[n,:])
      # ran=np.random.randint(0, high=25, size=1)
      # ran=ran.astype(int)
      pd=np.array([int(i) for i in copy.deepcopy(pd_o[0:25])]) 
      pd_p=copy.deepcopy(pd[:-1])
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
                  Loss_Matrix[n,0]=rni
                  Loss_Matrix[n,1]=pd[-1]
                  
            elif is_number(exp_output2ni) :
                  rni=int(exp_output1ni+exp_output2ni)
                  Loss_Per_Datani=abs(pd[-1]-rni)
                  Loss_Matrix[n,0]=rni
                  Loss_Matrix[n,1]=pd[-1]
            else:
                  Loss_Per_Datani = abs(pd[-1])
                  Loss_Matrix[n,0]=0
                  Loss_Matrix[n,1]=pd[-1]

      elif is_number(exp_output1ni) and is_number(exp_output2ni) :#####
            rni=int(exp_output1ni+exp_output2ni)
            Loss_Per_Datani=abs(pd[-1]-rni)
            Loss_Matrix[n,0]=rni
            Loss_Matrix[n,1]=pd[-1]

      elif is_number(exp_output1ni):
            rni=int(exp_output1ni)
            Loss_Per_Datani=abs(pd[-1]-rni)
            Loss_Matrix[n,0]=rni
            Loss_Matrix[n,1]=pd[-1]

      else:
            Loss_Per_Datani = abs(pd[-1])
            Loss_Matrix[n,0]=0
            Loss_Matrix[n,1]=pd[-1]
np.save('/home/rahul/ICML_IntGPT/find_model/new_data/Loss_25_sq5f',Loss_Matrix)
np.save('/home/rahul/ICML_IntGPT/find_model/new_data/Data_25_sq5f',Data_Matrix)


