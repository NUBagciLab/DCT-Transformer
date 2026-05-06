# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:33:32 2023

@author: pky0507
"""

# import numpy as np
# M = 8
# N = 8
# x = np.zeros((M, N))
# p = 0
# q = 0
# if p == 0:
#     ap = 1/np.sqrt(2)
# else:
#     ap = 1
# if q == 0:
#     aq = 1/np.sqrt(2)
# else:
#     aq = 1
# for m in range(M):
#     for n in range(N):
#         x[m, n] = 1/4*ap*aq*np.cos(np.pi*(2*m+1)*p/(2*M))*np.cos(np.pi*(2*n+1)*q/(2*N))
        
import matplotlib.pyplot as plt      
from scipy.fftpack import fft, dct, idct
import numpy as np
from scipy.linalg import toeplitz
D = dct(np.eye(8), axis=-1, norm="ortho")
D1 = idct(np.eye(8), axis=-1, norm="ortho")
N = 128
Y = abs(fft(D, n=N, axis=0))
x = np.zeros(8)
for i in range(8):
    x[i] = 0.9**i
P = toeplitz(x)
E, V = np.linalg.eig(P)
for i in range(8):
    if V[0, i]*D[0, i] < 0:
        V[:, i] = -V[:, i]


for i in range(8):
    plt.figure(figsize=(8, 8), dpi=300)
    plt.step(np.arange(9), np.insert(D[:, i], 0, D[0, i]), linewidth=3, label="DCT")
    plt.step(np.arange(9), np.insert(V[:, i], 0, V[0, i]), '--', linewidth=3, label="KLT")
    plt.legend(loc='lower left')
    plt.grid()
    plt.xlim([0, 8])
    plt.xticks(np.arange(9))
    plt.ylim([-.5, .5])
    #plt.title('D['+str(i)+']')
    plt.rcParams.update({'font.size': 30})
    plt.savefig('D'+str(i)+'.pdf', format="pdf", bbox_inches="tight")
    plt.show()

for i in range(8):
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(np.arange(N//2+1)/(N//2), Y[0:N//2+1, i], linewidth=3)
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 3])
    plt.xlabel(r'Frequency ($\times\pi$)', fontsize=30)
    plt.ylabel(r'Magnitude Response', fontsize=30)
    #plt.title('D['+str(i)+']')
    plt.rcParams.update({'font.size': 30})
    plt.savefig('F'+str(i)+'.pdf', format="pdf", bbox_inches="tight")
    plt.show()

plt.figure(figsize=(8, 8), dpi=300)
for i in range(8):
    plt.plot(np.arange(N//2+1)/(N//2), Y[0:N//2+1, i], linewidth=3)

plt.plot(np.arange(N//2+1)/(N//2), np.sqrt((Y[0:N//2+1, :]**2).sum(axis=-1)), '--', linewidth=3, color='black')
plt.grid()
plt.xlim([0, 1])
plt.ylim([0, 3])
plt.xlabel(r'Frequency ($\times\pi$)', fontsize=30)
plt.ylabel(r'Magnitude Response', fontsize=30)
#plt.title('D['+str(i)+']')
plt.rcParams.update({'font.size': 30})
plt.savefig('F.pdf', format="pdf", bbox_inches="tight")
plt.show()