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
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(db))
P = np.zeros(n_p)
for i in range(0,int(len(p)/len(db))):
    P[i] = p[i*len(db)]

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

data = pd.read_csv('result_p_b2.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(db))
P = np.zeros(n_p)
for i in range(0,int(len(p)/len(db))):
    P[i] = p[i*len(db)]

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
plt.ylabel("$s_u$")
plt.title("Contour plot of ESS vs $\Delta b$ and survival of waiters")

###################################
data = pd.read_csv('result_p_s.csv')
ds = np.array(data.loc[:,"d_s"])
n_ds = int(np.sqrt(len(ds)))
ds = ds[0:n_ds]


p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(ds))
P = np.zeros(n_p)
for i in range(0,n_p):
    P[i] = p[i*len(ds)]

h = np.zeros((n_p,n_ds))
k = 0


for i in range(0,n_p):
    for j in range(0,n_ds):
        h[i,j] = h_data[k]
        k+=1

fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(ds,P,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("P")
plt.title("Contour plot of ESS vs $\Delta s$ and probability of establishment")

##############################

data = pd.read_csv('result_p_sj.csv')
dsj = np.array(data.loc[:,"d_sj"])
n_dsj = int(np.sqrt(len(dsj)))
dsj = dsj[0:n_dsj]


p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(dsj))
P = np.zeros(n_p)
for i in range(0,n_p):
    P[i] = p[i*len(dsj)]
print(n_p)
h = np.zeros((n_p,n_dsj))
k = 0


for i in range(0,n_p):
    for j in range(0,n_dsj):
        h[i,j] = h_data[k]
        k+=1

fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(dsj,P,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s_j$')
plt.ylabel("P")
plt.title("Contour plot of ESS vs $\Delta s_j$ and probability of establishment")