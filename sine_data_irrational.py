def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import copy
    random.seed(10)
    np.random.seed(10)
    # sampling rate
    sr = 80
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,2,ts)
    from numpy.fft import fft, ifft

    sign_list=[-1,1]
    o_max_amp=np.linspace(-800,800,1601)
    # print(o_max_amp[800])
    max_amp=np.delete(o_max_amp,800)
    # print(max_amp)
    f1_o=np.sqrt(2)
    f2_o=np.sqrt(5)
    f3_o=np.sqrt(7)
    binary_list_pure=[(1,0,0),(0,1,0),(0,0,1)]
    binary_list_composition=[(1,1,0),(1,0,1),(0,1,1)]

    data_pure=[]
    for index in range(3):
        f1=copy.deepcopy(f1_o)*binary_list_pure[index][0]
        f2=copy.deepcopy(f2_o)*binary_list_pure[index][1]
        f3=copy.deepcopy(f3_o)*binary_list_pure[index][2]
        for max_amp1 in max_amp:
            # print("max amp", max_amp1)
            # print("f1", f1)
            # print("f2", f2)
            # print("f3", f3)
            a1=np.random.uniform(low=0, high=max_amp1)
            max_amp2=max_amp1-a1
            a2=np.random.uniform(low=0, high=max_amp2)
            max_amp3=max_amp1-a1-a2
            a3=np.random.uniform(low=0, high=max_amp3)
            # print("amplitude of a3")
            total_amplitude = a1+a2+a3
            # print(total_amplitude)
            total_amplitude=abs(total_amplitude)
            freq1 = f1
            h_t=np.random.uniform(low=0, high=800-total_amplitude, size=1)
            h_t=random.choice(sign_list)*h_t
            # print("vertical shift", h_t)
            
            
            x = a1*np.sin(2*np.pi*freq1*t)

            freq2 = f2
            x += a2*np.sin(2*np.pi*freq2*t)

            freq3 = f3
            x += a3* np.sin(2*np.pi*freq3*t )
            x += h_t
            x = np.rint(x)
            data_pure.append(x)

            # plt.figure(figsize = (4, 4))
            # plt.plot(t, x, 'r')
            # plt.ylabel('Amplitude')

            # plt.show()
            # plt.clf()
            # print(t.shape)
            # print("amplitude", str(a1), str(a2), str(a3))
            # print("range of the whole sequence", np.min(x), np.max(x))
            # print("value at 50th step",x[49])
            # print("value at 100th step",x[99])

            # print("print vertical shift", h_t)
            # X = fft(x[0:100])
            # N = len(X)
            # n = np.arange(N)
            # T = N/sr
            # freq = n/T

            # plt.figure(figsize = (12, 6))
            # plt.subplot(121)

            # plt.stem(freq, np.abs(X), 'b', \
            #         markerfmt=" ", basefmt="-b")
            # plt.xlabel('Freq (Hz)')
            # plt.ylabel('FFT Amplitude |X(freq)|')
            # plt.title('Length' + str(50))
            # plt.xlim(0, 15)
            # plt.show()
    # sampling rate
    sr = 80
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,2,ts)
    from numpy.fft import fft, ifft

    sign_list=[-1,1]
    o_max_amp=np.linspace(-800,800,1601)
    # print(o_max_amp)
    max_amp=np.delete(o_max_amp,800)
    # print(max_amp)
    f1_o=2
    f2_o=4
    f3_o=8
    binary_list_pure=[(1,0,0),(0,1,0),(0,0,1)]
    binary_list_composition=[(1,1,0),(1,0,1),(0,1,1)]

    data_composition=[]
    for index in range(3):
        f1=copy.deepcopy(f1_o)*binary_list_composition[index][0]
        f2=copy.deepcopy(f2_o)*binary_list_composition[index][1]
        f3=copy.deepcopy(f3_o)*binary_list_composition[index][2]
        for max_amp1 in max_amp:
            # print("max amp", max_amp1)
            # print("f1", f1)
            # print("f2", f2)
            # print("f3", f3)
            a1=np.random.uniform(low=0, high=max_amp1)
            max_amp2=max_amp1-a1
            a2=np.random.uniform(low=0, high=max_amp2)
            max_amp3=max_amp1-a1-a2
            a3=np.random.uniform(low=0, high=max_amp3)
            # print("amplitude of a3")
            total_amplitude = a1+a2+a3
            # print(total_amplitude)
            total_amplitude=abs(total_amplitude)
            freq1 = f1
            h_t=np.random.uniform(low=0, high=800-total_amplitude, size=1)
            h_t=random.choice(sign_list)*h_t
            # print("vertical shift", h_t)
            
            x = a1*np.sin(2*np.pi*freq1*t)

            freq2 = f2
            x += a2*np.sin(2*np.pi*freq2*t)

            freq3 = f3
            x += a3* np.sin(2*np.pi*freq3*t )
            x += h_t
            x = np.rint(x)
            data_composition.append(x)
            # plt.figure(figsize = (4, 4))
            # plt.plot(t, x, 'r')
            # plt.ylabel('Amplitude')

            # plt.show()
            # plt.clf()
            # print(t.shape)
            # print("amplitude", str(a1), str(a2), str(a3))
            # print("range of the whole sequence", np.min(x), np.max(x))
            # print("value at 50th step",x[49])
            # print("value at 100th step",x[99])
            # print("value at 150th step",x[149])
            # print("value at 200th step",x[199])
            # print("print vertical shift", h_t)
            # X = fft(x)
            # N = len(X)
            # n = np.arange(N)
            # T = N/sr
            # freq = n/T

            # plt.figure(figsize = (4, 4))
            # plt.subplot(121)

            # plt.stem(freq, np.abs(X), 'b', \
            #         markerfmt=" ", basefmt="-b")
            # plt.xlabel('Freq (Hz)')
            # plt.ylabel('FFT Amplitude |X(freq)|')
            # plt.title('Length' + str(50))
            # plt.xlim(0, 9)
            # plt.show()
    data_pure=np.array(data_pure)
    data_composition=np.array(data_composition)
    return data_pure, data_composition