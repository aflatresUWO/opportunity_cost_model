# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:43:15 2022

@author: aflatres
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('result_p_b.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
n_p = int(np.sqrt(len(p)))
P = np.zeros(n_p)
for i in range(0,int(np.sqrt(len(p)))):
    P[i] = p[i*len(db)]
print(P)
h_data = np.array(data.loc[:,"h_x"])

h = np.zeros((n_p,n_db))
k = 0

for i in range(0,n_p):
    for j in range(0,len(db)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(db,P,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$')
plt.ylabel("P")
plt.title("Contour plot of ESS vs $\Delta b$ and probability of establishment")

###################################
data = pd.read_csv('result_p_s.csv')
ds = np.array(data.loc[:,"d_s"])
n_ds = int(np.sqrt(len(ds)))
ds = ds[0:n_ds]

p = np.array(data.loc[:,"P"])
n_p = int(np.sqrt(len(p)))
P = np.zeros(n_p)
for i in range(0,int(np.sqrt(len(p)))):
    P[i] = p[i*len(ds)]

h_data = np.array(data.loc[:,"h_x"])

h = np.zeros((n_p,len(ds)))
k = 0

for i in range(0,n_p):
    for j in range(0,len(ds)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(ds,P,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("P")
#plt.title("Contour plot of ESS vs $\Delta b$ and probability of establishment")
#----------
data = pd.read_csv('result_p_sj.csv')
dsj = np.array(data.loc[:,"d_sj"])
dsj = ds[0:10]

p = np.array(data.loc[:,"P"])
n_p = int(np.sqrt(len(p)))
P = np.zeros(n_p)

for i in range(0,int(np.sqrt(len(p)))):
    P[i] = p[i*len(dsj)]

h_data = np.array(data.loc[:,"h_x"])

h = np.zeros((n_p,len(dsj)))
k = 0

for i in range(0,n_p):
    for j in range(0,len(dsj)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(dsj,P,h)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s_j$')
plt.ylabel("P")
#plt.title("Contour plot of ESS vs $\Delta b$ and probability of establishment")
#######################

data = pd.read_csv('result_pj_sj.csv')
dsj = np.array(data.loc[:,"d_sj"])
dsj = ds[0:10]
pj = np.array(data.loc[:,"Pj"])
n_pj = int(np.sqrt(len(pj)))
Pj = np.zeros(n_pj)

for i in range(0,int(np.sqrt(len(pj)))):
    Pj[i] = pj[i*len(dsj)]

h_data = np.array(data.loc[:,"h_x"])

h = np.zeros((n_pj,len(dsj)))
k = 0

for i in range(0,n_pj):
    for j in range(0,len(dsj)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(dsj,Pj,h)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s_j$')
plt.ylabel("p_j")
#plt.title("Contour plot of ESS vs $\Delta b$ and probability of establishment")

plt.show()