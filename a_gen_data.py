import numpy as np
def exponential(x, a, s, b):
  return a * np.exp(s * x) + b
test_count=-1
train_count=-1
train_data = np.zeros((49152,200))
test_data = np.zeros((4928,200))
#################### sine
frequency=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
train_freq=np.zeros((768*8, 200))
test_freq=np.zeros((77*8, 200))
num_train=0
num_test=0
for f in frequency:
    for repeat in range(845):
        p=np.pi*np.random.uniform(0,1,1)
        x = np.linspace(0, 2*np.pi*f, 200)
        a=200
        y_sine = a*np.sin(x+p)
        y_sine = np.rint(y_sine)
        if repeat<768:
            train_freq[num_train,:] = y_sine
            train_count+=1
            num_train+=1
            train_data[train_count] = y_sine
        else:
            test_freq[num_test,:] = y_sine
            test_count+=1
            num_test+=1
            test_data[test_count] = y_sine
# np.save('/home/rahul/LLM_Research/Data/data/freq_train', train_freq)
# np.save('/home/rahul/LLM_Research/Data/data/freq_test', test_freq)

amplitude=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
train_amp=np.zeros((768*8, 200))
test_amp=np.zeros((77*8, 200))
num_train=0
num_test=0
for a in amplitude:
    for repeat in range(845):
        f=1
        p=np.pi*np.random.uniform(0,1,1)
        x = np.linspace(0, 2*np.pi*f, 200)
        y_sine = a*np.sin(x+p)
        y_sine = np.rint(y_sine)
        if repeat<768:
            train_amp[num_train,:] = y_sine
            num_train+=1
            train_count+=1
            train_data[train_count] = y_sine
        else:
            test_amp[num_test,:] = y_sine
            num_test+=1
            test_count+=1
            test_data[test_count] = y_sine
# np.save('/home/rahul/LLM_Research/Data/data/amp_train', train_amp)
# np.save('/home/rahul/LLM_Research/Data/data/amp_test', test_amp)

############################### exponential

train_exp_u_e=np.zeros((768*4, 200))
test_amp_u_e=np.zeros((77*4, 200))
slope=[2, 3, 4, 5]
num_train_u_e=0
num_test_u_e=0
for s in slope:
    for repeat in range(845):
        x_values = np.linspace(0, 1, num=200)
        a=np.random.uniform(.75,1.25,1)
        b=np.random.uniform(-50,50,1)
        s=s
        y_values = exponential(x_values,a=a,s=s,b=b) 
        y_values = np.rint(y_values)
        if repeat<768:
            train_exp_u_e[num_train_u_e,:] = y_values
            num_train_u_e+=1
            train_count+=1
            train_data[train_count] = y_values
        else:
            test_amp_u_e[num_test_u_e,:] = y_values
            num_test_u_e+=1
            test_count+=1
            test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_exp_u_e', train_exp_u_e)
# np.save('/home/rahul/LLM_Research/Data/data/test_exp_u_e', test_amp_u_e)


train_exp_d_e=np.zeros((768*4, 200))
test_amp_d_e=np.zeros((77*4, 200))
slope=[2, 3, 4, 5]
num_train_d_e=0
num_test_d_e=0
for s in slope:
    for repeat in range(845):
        x_values = np.linspace(0, 1, num=200)
        a=-1*np.random.uniform(.75,1.25,1)
        b=np.random.uniform(-50,50,1)
        s=s
        y_values = exponential(x_values,a=a,s=s,b=b) 
        y_values = np.rint(y_values)
        if repeat<768:
            train_exp_d_e[num_train_d_e,:] = y_values
            num_train_d_e+=1
            train_count+=1
            train_data[train_count] = y_values
        else:
            test_amp_d_e[num_test_d_e,:] = y_values
            num_test_d_e+=1
            test_count+=1
            test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_exp_d_e', train_exp_d_e)
# np.save('/home/rahul/LLM_Research/Data/data/test_exp_d_e', test_amp_d_e)


train_exp_d_d=np.zeros((768*4, 200))
test_amp_d_d=np.zeros((77*4, 200))
slope=[2, 3, 4, 5]
num_train_d_d=0
num_test_d_d=0
for s in slope:
    for repeat in range(845):
        x_values = np.linspace(0, 1, num=200)
        a=np.random.uniform(.75,1.25,1)
        b=np.random.uniform(-50,50,1)
        s=s
        y_values = exponential(x_values,a=a,s=s,b=b) 
        y_values = np.rint(y_values)
        y_values = np.flip(y_values)
        if repeat<768:
            train_exp_d_d[num_train_d_d,:] = y_values
            num_train_d_d+=1
            train_count+=1
            train_data[train_count] = y_values
        else:
            test_amp_d_d[num_test_d_d,:] = y_values
            num_test_d_d+=1
            test_count+=1
            test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_exp_d_e', train_exp_d_d)
# np.save('/home/rahul/LLM_Research/Data/data/test_exp_d_e', test_amp_d_d)


train_exp_u_d=np.zeros((768*4, 200))
test_amp_u_d=np.zeros((77*4, 200))
slope=[2, 3, 4, 5]
num_train_u_d=0
num_test_u_d=0
for s in slope:
    for repeat in range(845):
        x_values = np.linspace(0, 1, num=200)
        a=-1*np.random.uniform(.75,1.25,1)
        b=np.random.uniform(-50,50,1)
        s=s
        y_values = exponential(x_values,a=a,s=s,b=b) 
        y_values = np.rint(y_values)
        y_values = np.flip(y_values)
        if repeat<768:
            train_exp_u_d[num_train_u_d,:] = y_values
            num_train_u_d+=1
            train_count+=1
            train_data[train_count] = y_values
        else:
            test_amp_u_d[num_test_u_d,:] = y_values
            num_test_u_d+=1
            test_count+=1
            test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_exp_u_d', train_exp_u_d)
# np.save('/home/rahul/LLM_Research/Data/data/test_exp_u_d', test_amp_u_d)

########################################################## mixed
train_mix_e_p=np.zeros((768*16, 200))
test_mix_e_p=np.zeros((77*16, 200))
frequency=[2.0, 5.0, 10.0, 20.0]
slope=[2, 3, 4, 5]
num_train_e_p=0
num_test_e_p=0
for f in frequency:
    p=np.pi*np.random.uniform(0,1,1)
    x = np.linspace(0, 2*np.pi*f, 200)
    a_s=1
    y_sine = a_s*np.sin(x+p)
    for s in slope:
        for repeat in range(845):
            x_values = np.linspace(0, 1, num=200)
            a=np.random.uniform(.75,1.25,1)
            b=np.random.uniform(-50,50,1)
            s=s
            y_values = exponential(x_values,a=a,s=s,b=b) * y_sine
            y_values = np.rint(y_values)
            if repeat<768:
                train_mix_e_p[num_train_e_p,:] = y_values
                num_train_e_p+=1
                train_count+=1
                train_data[train_count] = y_values
            else:
                test_mix_e_p[num_test_e_p,:] = y_values
                num_test_u_e+=1
                test_count+=1
                test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_exploding_periodic', train_mix_e_p)
# np.save('/home/rahul/LLM_Research/Data/data/test_exploding_periodic', test_mix_e_p)

train_mix_d_p=np.zeros((768*16, 200))
test_mix_d_p=np.zeros((77*16, 200))
frequency=[2.0, 5.0, 10.0, 20.0]
slope=[2, 3, 4, 5]
num_train_d_p=0
num_test_d_p=0
for f in frequency:
    p=np.pi*np.random.uniform(0,1,1)
    x = np.linspace(0, 2*np.pi*f, 200)
    a_s=1
    y_sine = a_s*np.sin(x+p)
    for s in slope:
        for repeat in range(845):
            x_values = np.linspace(0, 1, num=200)
            a=np.random.uniform(.75,1.25,1)
            b=np.random.uniform(-50,50,1)
            s=s
            y_values = exponential(x_values,a=a,s=s,b=b) * y_sine
            y_values = np.rint(y_values)
            y_values = np.flip(y_values)
            if repeat<768:
                train_mix_d_p[num_train_d_p,:] = y_values
                num_train_d_p+=1
                train_count+=1
                train_data[train_count] = y_values
            else:
                test_mix_d_p[num_test_d_p,:] = y_values
                num_test_d_p+=1
                test_count+=1
                test_data[test_count] = y_values
# np.save('/home/rahul/LLM_Research/Data/data/train_decaying_periodic', train_mix_d_p)
# np.save('/home/rahul/LLM_Research/Data/data/test_decaying_periodic', test_mix_d_p)
print("train_count",train_count)
print("test_count",test_count)
np.save("/home/rahul/LLM_Research/Data/data/train", train_data)
np.save("/home/rahul/LLM_Research/Data/data/test", test_data)

            
