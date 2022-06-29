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
plt.xlabel('Fecundity benefits')
plt.ylabel("Probability of establishment")
plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment")

###################

data = pd.read_csv('result_p_b.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_y"])
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
plt.title("Contour plot of ESS h_y vs $\Delta b$ and probability of establishment")

#################
p1 = np.array(data.loc[:,"P"])
p1 = p1[p1>0]
p2 = np.array(data.loc[:,"P2"])
p2 = p2[p2>0]
n_p = int(len(p1)/len(db))


P1 = np.zeros((n_p,n_db))
P2 = np.zeros((n_p,n_db))
k=0
for i in range(0,n_p):
    for j in range(0,n_db):
        P1[i,j] = p1[k]
        P2[i,j] = p2[k]
        k+=1
   
fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(db,P,(P2-P1)/P1)
fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$')
plt.ylabel("P before evolution")
plt.title("Contour plot of $\Delta P$ vs $\Delta b$ and probability of establishment before evo")

fig, ax=plt.subplots()

plt.scatter(P2,h)
plt.ylim(0.5,1)
######################
fig, ax=plt.subplots()
p = np.array(data.loc[:,"P"])
h= np.array(data.loc[:,"h_x"])
b= np.array(data.loc[:,"d_b"])

n_p = 20


P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,int(len(p)/len(db))):
    P1[i] = p[10+i*len(db)]
    H1[i] = h[10+i*len(db)]
    P2[i] = p[14+i*len(db)]
    H2[i] = h[14+i*len(db)]
    P3[i] = p[18+i*len(db)]
    H3[i] = h[18+i*len(db)]
P1 = P1[P1>0]
H1 = H1[H1>0]
P2 = P2[P2>0]
H2 = H2[H2>0]
P3 = P3[P3>0]
H3 = H3[H3>0]
plt.scatter(P1,H1)
plt.scatter(P2,H2)
plt.scatter(P3,H3)

plt.xlabel('Probability of establishment')
plt.ylabel("$h_x$")
plt.legend(["$d_b=0.47$","$d_b=0.68$","$d_b=0.89$"])
plt.title("Cooperation level $h_x$ value vs probability of establishment")

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
    P[i] = p[i*n_ds]

h = np.zeros((n_p,n_ds))

k = 0
for i in range(0,n_p):
    for j in range(0,n_ds):
        h[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(0,1,20)
cp=ax.contourf(ds,P,h)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("P")
plt.title("Contour plot of ESS  h_x vs $\Delta s$ and probability of establishment ")

##############################

data = pd.read_csv('result_p_s.csv')
ds = np.array(data.loc[:,"d_s"])
n_ds = int(np.sqrt(len(ds)))
ds = ds[0:n_ds]


p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_y"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(ds))
P = np.zeros(n_p)
for i in range(0,n_p):
    P[i] = p[i*n_ds]

h = np.zeros((n_p,n_ds))

k = 0
for i in range(0,n_p):
    for j in range(0,n_ds):
        h[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(ds,P,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("P")
plt.title("Contour plot of ESS h_y vs $\Delta s$ and probability of establishment ")

##############################
p1 = np.array(data.loc[:,"P"])
p1 = p1[p1>0]
p2 = np.array(data.loc[:,"P2"])
p2 = p2[p2>0]
n_p = int(len(p1)/len(ds))


P1 = np.zeros((n_p,n_ds))
P2 = np.zeros((n_p,n_ds))
k=0
for i in range(0,n_p):
    for j in range(0,n_ds):
        P1[i,j] = p1[k]
        P2[i,j] = p2[k]
        k+=1

fig, ax=plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(ds,P,(P2-P1)/P1)
fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("P before evolution")
plt.title("Contour plot of $\Delta P$ vs $\Delta s$ and probability of establishment before evo")
#############################

fig, ax=plt.subplots()
p = np.array(data.loc[:,"P"])
h= np.array(data.loc[:,"h_x"])
ds= np.array(data.loc[:,"d_s"])
print(ds)
n_p = 20


P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,int(len(p)/len(ds))):
    P1[i] = p[10+i*len(ds)]
    H1[i] = h[10+i*len(ds)]
    P2[i] = p[14+i*len(ds)]
    H2[i] = h[14+i*len(ds)]
    P3[i] = p[18+i*len(ds)]
    H3[i] = h[18+i*len(ds)]
print(P1)
P1 = P1[P1>0]
H1 = H1[H1>0]
P2 = P2[P2>0]
H2 = H2[H2>0]
P3 = P3[P3>0]
H3 = H3[H3>0]
plt.scatter(P1,H1)
plt.scatter(P2,H2)
plt.scatter(P3,H3)

plt.xlabel('Probability of establishment')
plt.ylabel("$h_x$")
plt.legend(["$d_b=0.47$","$d_b=0.68$","$d_b=0.89$"])
plt.title("Cooperation level $h_x$ value vs probability of establishment")


#####################

data = pd.read_csv('result_phi_bx.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]



phi = np.array(data.loc[:,"phi"])
h_data = np.array(data.loc[:,"h_x"])
print(phi)

n_phi = int(len(phi)/len(db))
Phi = np.zeros(n_phi)
for i in range(0,n_phi):
    Phi[i] = phi[i*n_db]
print(Phi)
h = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(0,1,10)
cp=ax.contourf(db,Phi,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("Phi")
plt.title("Contour plot of ESS h_y vs $\Delta s$ and probability of establishment ")


print("bit")



##############################
"""
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
plt.title("Contour plot of ESS h_x vs $\Delta s_j$ and probability of establishment")
############################


data = pd.read_csv('result_p_sj.csv')
dsj = np.array(data.loc[:,"d_sj"])
n_dsj = int(np.sqrt(len(dsj)))
dsj = dsj[0:n_dsj]


p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_y"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(dsj))
P = np.zeros(n_p)
for i in range(0,n_p):
    P[i] = p[i*len(dsj)]

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
plt.title("Contour plot of ESS h_y vs $\Delta s_j$ and probability of establishment")
############################

p1 = np.array(data.loc[:,"P"])
p1 = p1[p1>0]
p2 = np.array(data.loc[:,"P2"])
p2 = p2[p2>0]
n_p = int(len(p1)/len(ds))


P1 = np.zeros((n_p,n_dsj))
P2 = np.zeros((n_p,n_dsj))
k=0
for i in range(0,n_p):
    for j in range(0,n_dsj):
        P1[i,j] = p1[k]
        P2[i,j] = p2[k]
        k+=1
P = np.zeros(n_p)
for i in range(0,n_p):
    P[i] = p1[i*n_dsj]
fig, ax=plt.subplots()

cp=ax.contourf(dsj,P,(P2-P1)/P1)

fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$\Delta sj$')
plt.ylabel("P2")
plt.title("Contour plot of ESS vs $\Delta b$ and probability of establishment after evo")
"""
#########################################
data = pd.read_csv('time_spent_x.csv')
sx = np.array(data.loc[:,"sx"])
sy = np.array(data.loc[:,"sy"])
X0 = np.array(data.loc[:,"X0"])
X1 = np.array(data.loc[:,"X1"])
Y = np.array(data.loc[:,"Y"])
n_sx = int(np.sqrt(len(sx)))
sx = sx[0:n_sx]

s_y = np.zeros(n_sx)
for i in range(0,n_sx):
    s_y[i] = sy[i*n_sx]
    
n_sy = len(s_y)
k = 0

x0 = np.zeros((n_sx,n_sy))
x1 = np.zeros((n_sx,n_sy))
y = np.zeros((n_sx,n_sy))
for i in range(0,n_sx):
    for j in range(0,n_sy):
        x0[i,j] = X0[k]
        x1[i,j] = X1[k]
        y[i,j] = Y[k]
        k+=1
breeder = x1+y
fig, ax=plt.subplots()

cp=ax.contourf(sx,s_y,x0)

fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$s_x$')
plt.ylabel("$s_y$")
plt.title("Time spent as X")

fig, ax=plt.subplots()

cp=ax.contourf(sx,s_y,x1)

fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$s_x$')
plt.ylabel("$s_y$")
plt.title("Time spent as X after evolution")
plt.plot(sx,sx,"red")
    
fig, ax=plt.subplots()
level = np.linspace(0,30,15)
cp=ax.contourf(sx,s_y,y,levels = level)

fig.colorbar(cp)#Add a colorbar to a plot
plt.plot(sx,sx,"red")

plt.xlabel('$s_x$')
plt.ylabel("$s_y$")
plt.title("Time spent as Y")

fig, ax=plt.subplots()
cp=ax.contourf(sx,s_y,breeder)
fig.colorbar(cp)#Add a colorbar to a plot


###########
##############



"""
############################
data = pd.read_csv('result_s_db.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
print(n_db)
db= db[0:n_db]
print(db)

s = np.array(data.loc[:,"s"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[s>0]
s=s[s>0]
n_s = int(len(s)/len(db))
s_u = np.zeros(n_s)
for i in range(0,n_s):
    s_u[i] = s[i*len(db)]
print(s_u)
h = np.zeros((n_s,n_db))
k = 0


for i in range(0,n_s):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1
print(db)
fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(db,s_u,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$')
plt.ylabel("$s_u$")
plt.title("Contour plot of ESS vs $\Delta b$ and survival of waiters")
#######################################
data = pd.read_csv('result_s_ds.csv')
ds = np.array(data.loc[:,"d_s"])
n_ds = int(np.sqrt(len(ds)))
print(n_ds)
ds= ds[0:n_ds]
print(ds)

s = np.array(data.loc[:,"s"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[s>0]
s=s[s>0]
n_s = int(len(s)/len(ds))
s_u = np.zeros(n_s)
for i in range(0,n_s):
    s_u[i] = s[i*len(ds)]
print(s_u)
h = np.zeros((n_s,n_ds))
k = 0


for i in range(0,n_s):
    for j in range(0,n_ds):
        h[i,j] = h_data[k]
        k+=1
print(db)
fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(ds,s_u,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta s$')
plt.ylabel("$s_u$")
plt.title("Contour plot of ESS vs $\Delta s$ and survival of waiters")
##############################

data = pd.read_csv('result_pju_db.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
print("caoutchouc")
print(n_db)
db= db[0:n_db]
print(db)

p = np.array(data.loc[:,"Pju"])
h_data = np.array(data.loc[:,"h_x"])


n_p = int(len(p)/len(db))
pju = np.zeros(n_p)
for i in range(0,n_p):
    pju[i] = p[i*len(db)]
print(pju)
h = np.zeros((n_p,n_db))
k = 0


for i in range(0,n_p):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1
print(db)
fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(db,pju,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$')
plt.ylabel("$\Delta_P$")
plt.title("Contour plot of ESS vs $\Delta P$ and fecundity of breeders")

################################### 

data = pd.read_csv('result_prime_db.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
print(n_db)
db= db[0:n_db]
print(db)

p = np.array(data.loc[:,"P1"])
h_data = np.array(data.loc[:,"h_x"])


n_p = int(len(p)/len(db))
pju = np.zeros(n_p)
for i in range(0,n_p):
    pju[i] = p[i*len(db)]
print(pju)
h = np.zeros((n_p,n_db))
k = 0


for i in range(0,n_p):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1
print(db)
fig, ax=plt.subplots()


level = np.linspace(0,1,10)
cp=ax.contourf(db,pju,h,levels = level)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$')
plt.ylabel("$P1$")
plt.title("Contour plot of ESS vs $\Delta P$ and fecundity of breeders")"""