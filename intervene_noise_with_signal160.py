
def main(data):
      # import sine_data 
      # import poly_data
      import matplotlib.pyplot as plt
      from nnsight import LanguageModel
      import matplotlib.pyplot as plt
      import numpy as np
      import torch
      import matplotlib.pyplot as plt
      import time
      import copy 
      start_time = time.time()
      def is_number(n):
            try:
                  float(n)  
            except ValueError:
                  return False
            return True
      np.random.seed(0)
      llm = LanguageModel("openai-community/gpt2", device_map="auto")
      sine_train_data = data
      print("sine_exp_train_data shape:\n", sine_train_data.shape)
      # print("range of sine_exp data", np.min(sine_exp_train_data), np.max(sine_exp_train_data))
      # poly_data_ = poly_data.main()
      # print("poly data shape:\n", poly_data_.shape)
      # print("range of poly data", np.min(poly_data_), np.max(poly_data_))
      mean = 0
      std = 1 
      num_samples = 160
      noise = np.rint(np.random.normal(mean, std, size=num_samples))
      nd=copy.deepcopy(noise)
      # test_data=np.zeros(((poly_data_.shape[0]+sine_exp_train_data.shape[0]),200))
      # test_data=np.zeros((poly_data_.shape[0],200))
      # test_data=poly_data_[(400+400+101):(400+400+400+101),:]
      # test_data=poly_data_[(400+101):(400+400+101),:]
      # print(test_data.shape)
      # test_data[0:poly_data_.shape[0],:]=poly_data_
      # test_data[poly_data_.shape[0]:,:]=sine_exp_train_data
      test_data=copy.deepcopy(sine_train_data)
      Loss_Per_Data=0
      Loss_Tensor = np.zeros((test_data.shape[0], 22, 13))
      
      Loss_gt_intervention = np.zeros((test_data.shape[0]))
      for data_index in range(test_data.shape[0]):
      # for data_index in range(1):
      # for data_index in range(101,101+800):
      # for data_index in range(0,101):
            s=0
            for steps in list([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,60,80,120,160]):
            # for steps in list([0]):
                  reference_signal=copy.deepcopy(test_data[data_index,:])
                  td=np.array([int(i) for i in copy.deepcopy(reference_signal[:-1])]) 
                  tds=', '.join(map(str, td))
                  td_s = tds + str(",")
                  
                  if steps == 0 :
                        nds=np.array([int(i) for i in copy.deepcopy(noise[:-1])]) 
                        nds=', '.join(map(str, nds))
                        nd_s = nds + str(",")
                        # print(nd_s)
                  else:
                        nd[-1-steps:-1]=copy.deepcopy(reference_signal[-1-steps:-1])
                        nds=np.array([int(i) for i in copy.deepcopy(nd[:-1])]) 
                        nds=', '.join(map(str, nds))
                        nd_s = nds + str(",")
                  nd_sni=nd_s
                  # print(nd_s)
                  with llm.trace(nd_sni) as tracer:
                        exp_token1ni = llm.lm_head.output.argmax(dim=-1).save()
                  nd_sni_sl = nd_sni + llm.tokenizer.decode(exp_token1ni[0][-1])
                  with llm.trace(nd_sni_sl) as tracer:
                        exp_token2ni = llm.lm_head.output.argmax(dim=-1).save()
                  nd_sni_sl_f= nd_sni_sl + llm.tokenizer.decode(exp_token2ni[0][-1])
                  with llm.trace(nd_sni_sl_f) as tracer:
                        exp_token3ni = llm.lm_head.output.argmax(dim=-1).save()
                  exp_output1ni=llm.tokenizer.decode(exp_token1ni[0][-1])
                  exp_output1ni=exp_output1ni.replace(" ", "")
                  exp_output2ni=llm.tokenizer.decode(exp_token2ni[0][-1])
                  exp_output2ni=exp_output2ni.replace(" ", "")
                  exp_output3ni=llm.tokenizer.decode(exp_token3ni[0][-1])
                  exp_output3ni=exp_output3ni.replace(" ", "")
########################## no intervention 
                  # print(exp_output1ni,exp_output2ni,exp_output3ni)
                  if exp_output1ni == '-':
                        if is_number(exp_output2ni) and is_number(exp_output3ni):
                              rni=int(exp_output1ni+exp_output2ni+exp_output3ni)
                              Loss_Per_Datani=abs(reference_signal[-1]-rni)
                              # print("this as well")

                              
                        elif is_number(exp_output2ni) :
                              rni=int(exp_output1ni+exp_output2ni)
                              # print("print rni", rni)
                              # print(reference_signal[-1])
                              Loss_Per_Datani=abs(reference_signal[-1]-rni)
                              # print("this")

                        else:
                              Loss_Per_Datani = abs(reference_signal[-1])
                              

                              
                  elif is_number(exp_output1ni) and is_number(exp_output2ni) :#####
                        rni=int(exp_output1ni+exp_output2ni)
                        Loss_Per_Datani=abs(reference_signal[-1]-rni)
                        
                        
                  elif is_number(exp_output1ni):
                        rni=int(exp_output1ni)
                        Loss_Per_Datani=abs(reference_signal[-1]-rni)
                        
                        
                  else:
                        Loss_Per_Datani = abs(reference_signal[-1])
                        



########################## no intervention 
                  
                  Loss_Tensor[data_index,s,0]=Loss_Per_Datani
                  # print(exp_output1ni,exp_output2ni,exp_output3ni)
                  # print("prior",Loss_Tensor[data_index,0,0])
                  for layer in range(12):
                  # for layer in range(0):
                        if steps == 0 :
                              nds=np.array([int(i) for i in copy.deepcopy(nd[:-1])]) 
                              nds=', '.join(map(str, nds))
                              nd_s = nds + str(",")
                        else:
                              nd[-1-steps:-1]=copy.deepcopy(reference_signal[-1-steps:-1])
                              nds=np.array([int(i) for i in copy.deepcopy(nd[:-1])]) 
                              nds=', '.join(map(str, nds))
                              nd_s = nds + str(",")
                        reference_signal=copy.deepcopy(test_data[data_index,:])
                        td=np.array([int(i) for i in copy.deepcopy(reference_signal[:-1])]) 
                        tds=', '.join(map(str, td))
                        td_s = tds + str(",")
                        
                        nd_s_=nd_s
                        past=-1
                        with llm.trace(td_s) as tracer:
                              exp_token1 = llm.lm_head.output.argmax(dim=-1).save()
                              hs = llm.transformer.h[layer].output[0][:, past, :].save()
                        td_sl = td_s + llm.tokenizer.decode(exp_token1[0][-1])
                        with llm.trace(td_sl) as tracer:
                              exp_token2 = llm.lm_head.output.argmax(dim=-1).save()
                              hs2 = llm.transformer.h[layer].output[0][:, past, :].save()
                        td_sf= td_sl + llm.tokenizer.decode(exp_token2[0][-1])
                        with llm.trace(td_sf) as tracer:
                              exp_token3 = llm.lm_head.output.argmax(dim=-1).save()
                              hs3 = llm.transformer.h[layer].output[0][:, past, :].save() 

                        # the edited model will now always predict "Paris" as the next token
                        with llm.edit() as llm_edited:
                              llm.transformer.h[layer].output[0][:, past, :] = hs
                        # llm.transformer.h[layer_num].mlp.output[0, -1, :] = noise_activation[layer_num,-1,:]
                        
                        # ...with the output of the edited model
                        with llm_edited.trace(nd_s_) as tracer:
                              modified_tokens1 = llm.lm_head.output.argmax(dim=-1).save()
                        with llm.edit() as llm_edited2:
                              llm.transformer.h[layer].output[0][:, past, :] = hs2

                        nd_s__ = nd_s_ + llm.tokenizer.decode(modified_tokens1[0][-1])
                        with llm_edited2.trace(nd_s__) as tracer:
                              modified_tokens2 = llm.lm_head.output.argmax(dim=-1).save()
                        nd_s___ = nd_s__ + llm.tokenizer.decode(modified_tokens2[0][-1])
                        with llm.edit() as llm_edited3:
                              llm.transformer.h[layer].output[0][:, past, :] = hs3
                        with llm_edited3.trace(nd_s___) as tracer:
                              modified_tokens3 = llm.lm_head.output.argmax(dim=-1).save()
                              
                        modified_output1=llm.tokenizer.decode(modified_tokens1[0][-1])
                        modified_output1=modified_output1.replace(" ", "")
                        modified_output2=llm.tokenizer.decode(modified_tokens2[0][-1])
                        modified_output2=modified_output2.replace(" ", "")
                        modified_output3=llm.tokenizer.decode(modified_tokens3[0][-1])
                        modified_output3=modified_output3.replace(" ", "")

            ###############################################################

                        exp_output1=llm.tokenizer.decode(exp_token1[0][-1])
                        exp_output1=exp_output1.replace(" ", "")
                        exp_output2=llm.tokenizer.decode(exp_token2[0][-1])
                        exp_output2=exp_output2.replace(" ", "")
                        exp_output3=llm.tokenizer.decode(exp_token3[0][-1])
                        exp_output3=exp_output3.replace(" ", "")
                        
                        if exp_output1 == '-':
                              if is_number(exp_output2) and is_number(exp_output3):
                                    gt=int(exp_output1+exp_output2+exp_output3)
                                    
                                    if modified_output1 == '-':
                                          if is_number(modified_output2) and is_number(modified_output3):
                                                Y=int(modified_output1+modified_output2+modified_output3)
                                                Loss_Per_Data = abs(gt-Y)
                                              
                                          elif is_number(modified_output2):
                                                Y=int(modified_output1+modified_output2)
                                                Loss_Per_Data = abs(gt-Y)
                                                
                                          else:
                                                Loss_Per_Data = abs(gt)
                                               
                                          
                                    elif is_number(modified_output1) and is_number(modified_output2):
                                          Y=int(modified_output1+modified_output2)
                                          Loss_Per_Data = abs(gt-Y)
                                          
                                    elif is_number(modified_output1):
                                          Y=int(modified_output1)
                                          Loss_Per_Data = abs(gt-Y)
                                        
                                    else:
                                          Loss_Per_Data = abs(gt)
                                          
                              elif is_number(exp_output2) :
                                    gt=int(exp_output1+exp_output2)
                                    
                                    if modified_output1 == '-':
                                          if is_number(modified_output2) and is_number(modified_output3):
                                                Y=int(modified_output1+modified_output2+modified_output3)
                                                Loss_Per_Data = abs(gt-Y)

                                          elif is_number(modified_output2):
                                                Y=int(modified_output1+modified_output2)
                                                Loss_Per_Data = abs(gt-Y)
                                          else:
                                                Loss_Per_Data = abs(gt)

                                    elif is_number(modified_output1) and is_number(modified_output2):
                                          Y=int(modified_output1+modified_output2)
                                          Loss_Per_Data = abs(gt-Y)
                                    elif is_number(modified_output1):
                                          Y=int(modified_output1)
                                          Loss_Per_Data = abs(gt-Y)
                                    else:
                                          Loss_Per_Data = abs(gt)

                              else:
                                    Loss_Per_Data = 0
                                    
                        elif is_number(exp_output1) and is_number(exp_output2) :#####
                              gt=int(exp_output1+exp_output2)
                              if modified_output1 == '-':
                                    if is_number(modified_output2) and is_number(modified_output3):
                                          Y=int(modified_output1+modified_output2+modified_output3)
                                          Loss_Per_Data = abs(gt-Y)
                                    elif is_number(modified_output2):
                                          Y=int(modified_output1+modified_output2)
                                          Loss_Per_Data = abs(gt-Y)
                                    else:
                                          Loss_Per_Data = abs(gt)
                              elif is_number(modified_output1) and is_number(modified_output2):
                                    Y=int(modified_output1+modified_output2)
                                    Loss_Per_Data = abs(gt-Y)
                              elif is_number(modified_output1):
                                    Y=int(modified_output1)
                                    Loss_Per_Data = abs(gt-Y)
                              else:
                                    Loss_Per_Data = abs(gt)                                                      
                        elif is_number(exp_output1):
                              gt=int(exp_output1)

                              if steps == 150:
                                    Loss_gt_intervention[data_index]=abs(reference_signal[-1]-gt)
                              if modified_output1 == '-':
                                    if is_number(modified_output2) and is_number(modified_output3):
                                          Y=int(modified_output1+modified_output2+modified_output3)
                                          Loss_Per_Data = abs(gt-Y)
                                    elif is_number(modified_output2):
                                          Y=int(modified_output1+modified_output2)
                                          Loss_Per_Data = abs(gt-Y)
                                    else:
                                          Loss_Per_Data = abs(gt)
                              elif is_number(modified_output1) and is_number(modified_output2):
                                    Y=int(modified_output1+modified_output2)
                                    Loss_Per_Data = abs(gt-Y)
                              elif is_number(modified_output1):
                                    Y=int(modified_output1)
                                    Loss_Per_Data = abs(gt-Y)
                              else:
                                    Loss_Per_Data = abs(gt)
                        else:
                              Loss_Per_Data = 0

            ###############################################################
                        Loss_Tensor[data_index,s,(layer+1)]=Loss_Per_Data
                        # print("later",Loss_Tensor[data_index,0,0])
                  # print(Loss_Tensor[data_index,:,:])
                  s+=1
                  
      print("--- %s seconds ---" % (time.time() - start_time))
      return Loss_Tensor
# main()





