# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:24:58 2019

@author: Vinícius Rezende Carvalho
"""
import numpy as np

def  VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    
    
    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    
    # 检查 f 的长度是否为偶数，如果不是，则添加一个零
    #if len(f) % 2 != 0:
    #    f = np.append(f, 0)  # 在数组末尾添加一个零
    if len(f) % 2:
        f = f[:-1]  # 检查信号数组f长度是否为偶数，若不是，则去掉最后一个元素保证其为偶数

    # Period and sampling frequency of input signal
    fs = 1./len(f)    #采样频率为信号长度的倒数
    
    ltemp = len(f)//2    #这行代码计算信号长度的一半，并取整。这个值用于帮助定义信号的前半部分和后半部分，为镜像操作做准备
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)     #原始[a,b,c,d]
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))  #镜像后[b,a,a,b,c,d,d,c]
#镜像是为了得到一个对称的信号做傅里叶
    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)  #经过镜像处理后的信号长度
    t = np.arange(1,T+1)/T  #标准化频率范围（0-1）
    
    # Spectral Domain discretization
    freqs = t-0.5-(1/T)   #计算频域的离散频率（-0.5,0.5）

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500  #迭代次数最大为500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones(K)
    
    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])


    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0
            
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0
    
    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
    
    # other inits
    uDiff = tol+np.spacing(1) # update step
    n = 0 # loop counter
    sum_uk = 0 # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)    

    #*** Main loop for iterative updates***

    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)
        
        # update first omega if not held at 0
        if not(DC):
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # update of any other mode
        for k in np.arange(1,K):
            #accumulator
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            # mode spectrum
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            # center frequencies
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
            
        # Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)
        
        # loop counter
        n = n+1
        
        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)        
            
    #Postprocessing and cleanup
    
    #discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # Signal reconstruction
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    # remove mirror part
    u = u[:,T//4:3*T//4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))

    return u, u_hat, omega
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = r'G:\SWH prediction\VMD-CNN-Transformer 代码及数据\1-MPCT\数据集\VMD\\NDBCSWH.xlsx'
df = pd.read_excel(file_path)
data = df['signal'].values

# 打印原始数据长度


# 假设您已经有一个名为 VMD 的函数来处理这些数据
# alpha, tau, K, DC, init, tol 是 VMD 函数的参数
alpha, tau, K, DC, init, tol = 2204, 0, 9, False, 1, 1e-7
u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

print("shape of u:", u.shape)
u_df = pd.DataFrame(u.T)  # Transpose to make each mode a column

# Save to a CSV file
u_df.to_csv(r'G:\SWH prediction\VMD-CNN-Transformer 代码及数据\1-MPCT\数据集\VMD\u_data.csv', index=False)
'''
# 绘图展示结果
plt.figure()
for i in range(u.shape[0]):
    plt.subplot(u.shape[0]+1, 1, i+2)
    plt.plot(u[i, :])
    plt.ylabel(f'IMF {i+1}')
plt.show()
'''
# 将 u 转换为 DataFrame
#u_df = pd.DataFrame(u.transpose(), columns=[f'IMF {i+1}' for i in range(u.shape[0])])
#u_df.to_csv('u_modes_760.csv', index=False)


# 创建实部和虚部的 DataFrame
#u_hat_real = pd.DataFrame(u_hat.real.transpose(), columns=[f'IMF {i+1} Real' for i in range(u_hat.shape[1])])
#u_hat_imag = pd.DataFrame(u_hat.imag.transpose(), columns=[f'IMF {i+1} Imag' for i in range(u_hat.shape[1])])

# 合并 DataFrame 并导出
#u_hat_df = pd.concat([u_hat_real, u_hat_imag], axis=1)
#u_hat_df.to_csv('u_hat.csv', index=False)


# 将 omega 转换为 DataFrame
#omega_df = pd.DataFrame(omega, columns=[f'Omega {i+1}' for i in range(omega.shape[1])])
#omega_df.to_csv('omega.csv', index=False)

