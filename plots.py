# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:43:15 2022

@author: aflatres
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
######################
#Probability of establishment (b) and fecundity benefits
data = pd.read_csv('result_p_b.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(db))
b = np.linspace(1,8,n_db)
b=b[n_db-n_p:]
s = np.linspace(0.1,0.99,n_db)
s=s[n_db-n_p:]
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

level = np.linspace(0,1,7)

cp=ax.contourf(db,b,h,levels = level)
level = np.linspace(0,0,1)
ax.contour(db,b,h,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()

##################

#Probability of establishment (b) and fecundity benefits
data = pd.read_csv('result_p_b_05.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))

db = db[0:n_db]

b = np.linspace(1,8,n_db)
p = np.array(data.loc[:,"P"])

h_data = np.array(data.loc[:,"h_x"])
p=p[p>0]


n_b = len(b)
n_p = int(len(p)/len(db))

h = np.zeros((n_b,n_db))

k = 0
print(b)
print(db)
for i in range(0,n_b):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1
h = h[n_db-n_p:,:]
b=b[n_db-n_p:]
fig, ax=plt.subplots()

level = np.linspace(0,1,7)
cp=ax.contourf(db,b,h,levels = level)
level = np.linspace(0,0,1)
ax.contour(db,b,h,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
##################
# Same but with GA
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
level = np.linspace(0,1,7)
b = np.linspace(1,8,n_db)
b=b[n_db-n_p:]
cp=ax.contourf(db,b,h,levels = level)
level = np.linspace(0,0,1)
ax.contour(db,b,h,levels = level,colors = "red")
fig.colorbar (cp,label = "$h_x$")
#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)

###################
"""
#Same as 1 but with h_y
data = pd.read_csv('result_p_b.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
hy_data = np.array(data.loc[:,"h_y"])
hy_data = hy_data[p>0]
hx_data = np.array(data.loc[:,"h_x"])
hx_data = hx_data[p>0]
p=p[p>0]
n_p = int(len(p)/len(db))
P = np.zeros(n_p)
for i in range(0,int(len(p)/len(db))):
    P[i] = p[i*len(db)]

hy = np.zeros((n_p,n_db))
k = 0
hx = np.zeros((n_p,n_db))


for i in range(0,n_p):
    for j in range(0,len(db)):
        hy[i,j] = hy_data[k]
        hx[i,j] = hx_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,0.25,7)
cp=ax.contourf(db,b,hx-hy,levels = level)
fig.colorbar (cp,label = "$h_x-h_y$")
level = np.linspace(0,0,1)
ax.contour(db,b,h,levels = level,colors = "red")
#plt.xlabel('$\Delta b$',fontsize = 15.0)
#plt.ylabel("P",fontsize = 15.0)
#plt.title("Contour plot of CSS h_y vs $\Delta b$ and probability of establishment")
"""
#################
#Comparison of P before and after evolution
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
cp=ax.contourf(db,P,(P2-P1))
fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$\Delta b$',fontsize = 15.0)
plt.ylabel("P before evolution",fontsize = 15.0)
plt.title("Contour plot of $\Delta P$ vs $\Delta b$ and probability of establishment before evo")



######################
#Probability of establishment and hx
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
    P1[i] = p[18+i*len(db)]
    H1[i] = h[18+i*len(db)]
    P2[i] = p[14+i*len(db)]
    H2[i] = h[14+i*len(db)]
    P3[i] = p[10+i*len(db)]
    H3[i] = h[10+i*len(db)]
P1 = P1[P1>0]
H1 = H1[H1>0]
P2 = P2[P2>0]
H2 = H2[H2>0]
P3 = P3[P3>0]
H3 = H3[H3>0]
plt.scatter(P1,H1)
plt.scatter(P2,H2)
plt.scatter(P3,H3)

plt.xlabel('Probability of establishment',fontsize = 20.0)
plt.ylabel("$h_x$",fontsize = 20.0)
plt.legend(["$\Delta b=0.89$","$\Delta b=0.68$","$\Delta b=0.47$"],fontsize =15.0)
#plt.title("Cooperation level $h_x$ value vs probability of establishment",fontsize = 15.0)
########################
#Actual levl of help

data = pd.read_csv('result_p_db_b.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]
b = np.array(data.loc[:,"bx"])
p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
b = b[p>0]
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
help_b = h_data*b
helpb = np.zeros((n_p,n_db))
k = 0

for i in range(0,n_p):
    for j in range(0,len(db)):
        helpb[i,j] = help_b[k]
        k+=1
fig, ax=plt.subplots()
helpb = 1-np.exp(-helpb)
level = np.linspace(0,1,10)
cp=ax.contourf(db,P,helpb)

fig.colorbar (cp)#Add a colorbar to a plot
plt.xlabel('Fecundity benefits',fontsize = 15.0)
plt.ylabel("Probability of establishment",fontsize = 15.0)
plt.title("Actual level of help")
###################################
#Probability of estb (bx,by) with ds
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
plt.xlabel('$\Delta s$',fontsize = 15.0)
plt.ylabel("P",fontsize = 15.0)
plt.title("Contour plot of CSS  h_x vs $\Delta s$ and probability of establishment ")

##############################
#Same as above but hy
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
plt.xlabel('$\Delta s$',fontsize = 15.0)
plt.ylabel("P",fontsize = 15.0)
plt.title("Contour plot of CSS h_y vs $\Delta s$ and probability of establishment ")

##############################
#change of p before and after evolution
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
plt.xlabel('$\Delta s$',fontsize = 15.0)
plt.ylabel("P before evolution",fontsize = 15.0)
plt.title("Contour plot of $\Delta P$ vs $\Delta s$ and probability of establishment before evo")
#############################
#p and hx level
fig, ax=plt.subplots()
p = np.array(data.loc[:,"P"])
h= np.array(data.loc[:,"h_x"])
ds= np.array(data.loc[:,"d_s"])
n_p = 20
n_ds = int(np.sqrt(len(ds)))
ds = ds[0:n_ds]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,int(len(p)/len(ds))):
    P1[i] = p[18+i*len(ds)]
    
    H1[i] = h[18+i*len(ds)]
    P2[i] = p[14+i*len(ds)]
    H2[i] = h[14+i*len(ds)]
    P3[i] = p[10+i*len(ds)]
    H3[i] = h[10+i*len(ds)]

P1 = P1[P1>0]
H1 = H1[H1>0]
P2 = P2[P2>0]
H2 = H2[H2>0]
P3 = P3[P3>0]
H3 = H3[H3>0]
plt.scatter(P1,H1)
plt.scatter(P2,H2)
plt.scatter(P3,H3)

plt.xlabel('Probability of establishment',fontsize = 20.0)
plt.ylabel("$h_x$",fontsize = 20.0)
plt.legend(["$\Delta s=0.89$","$\Delta s=0.68$","$ \Delta s=0.47$"])
#plt.title("Cooperation level $h_x$ value vs probability of establishment")


#####################
#Impact of GA and phi on coop
data = pd.read_csv('result_phi_bx_ga2.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]



phi = np.array(data.loc[:,"phi"])
h_data = np.array(data.loc[:,"h_x"])


n_phi = int(len(phi)/len(db))
Phi = np.zeros(n_phi)
for i in range(0,n_phi):
    Phi[i] = phi[i*n_db]
h = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        h[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(1,10,7)
level = np.log10(level)
cp=ax.contourf(db,Phi,h,levels= level)
fig.colorbar (cp)
level = np.linspace(0,0,1)
cp=ax.contour(db,Phi,h,levels= level, colors = "red")

#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("Inbreeding coeff $\Phi$",fontsize = 15.0)
#plt.title("Contour plot of CSS $h_x$ vs $\Delta b$ and $\Phi$ ",fontsize =15.0)



##############################
#impact of phi and ga on reproductive value
data = pd.read_csv('result_phi_bx_ga2.csv')
db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]



phi = np.array(data.loc[:,"phi"])
h_data = np.array(data.loc[:,"GA"])


n_phi = int(len(phi)/len(db))
Phi = np.zeros(n_phi)
for i in range(0,n_phi):
    Phi[i] = phi[i*n_db]
nu = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        nu[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(0,0.4,7)
cp=ax.contourf(db,Phi,nu,cmap = 'cividis')

fig.colorbar (cp)#Add a colorbar to a plot
level = np.linspace(0,0,1)
cp=ax.contour(db,Phi,h,levels= level, colors = "red")

plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("Inbreeding coeff $\Phi$",fontsize = 15.0)
#plt.title(" Effect of $\Phi$ on repro value ", fontsize = 15.0)


################################
#Time spent as X
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
plt.xlabel('$s_x$',fontsize = 20.0)
plt.ylabel("$s_y$",fontsize = 20.0)
plt.title("Time spent as X")

fig, ax=plt.subplots()

cp=ax.contourf(sx,s_y,x1)

fig.colorbar(cp)#Add a colorbar to a plot
plt.xlabel('$s_x$',fontsize = 20.0)
plt.ylabel("$s_y$",fontsize = 20.0)
plt.title("Time spent as X after evolution")
plt.plot(sx,sx,"red")
    
fig, ax=plt.subplots()
level = np.linspace(0,30,15)
cp=ax.contourf(sx,s_y,y)

fig.colorbar(cp)#Add a colorbar to a plot
plt.plot(sx,sx,"red")

plt.xlabel('$s_x$',fontsize = 20.0)
plt.ylabel("$s_y$",fontsize = 20.0)
plt.title("Time spent as Y")

fig, ax=plt.subplots()
cp=ax.contourf(sx,s_y,breeder)
fig.colorbar(cp)#Add a colorbar to a plot


###########
#When P is changing with s_x, s_y

data = pd.read_csv('result_p_sx.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]

p = np.delete(p,0)
p = np.delete(p,0)
p = np.delete(p,0)
h_data = np.delete(h_data,0)
h_data = np.delete(h_data,0)
h_data = np.delete(h_data,0)

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

level = np.linspace(0,1,7)

cp=ax.contourf(db,P,h,levels = level)
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
###################################
#When P is changing because of s_u
data = pd.read_csv('result_p_su.csv')

db = np.array(data.loc[:,"d_b"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

su = np.array(data.loc[:,"s_u"])
h_data = np.array(data.loc[:,"h_x"])



n_su = int(len(su)/len(db))
s_u = np.zeros(n_su)
for i in range(0,int(len(su)/len(db))):
    s_u[i] = su[i*len(db)]
h = np.zeros((n_su,n_db))
k = 0

for i in range(0,n_su):
    for j in range(0,len(db)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,7)

cp=ax.contourf(db,s_u,h,levels = level)
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("s_u",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
################################
#change of number of partners
data = pd.read_csv('result_M_db.csv')

db = np.array(data.loc[:,"db"])
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]

print(db)
h_data = np.array(data.loc[:,"h_x"])


M = np.array(data.loc[:,"M"])
n_M = int(len(M)/len(db))
m = np.zeros(n_M)
for i in range(0,int(len(M)/len(db))):
    m[i] = M[i*len(db)]
print(m)
h = np.zeros((n_M,n_db))
k = 0

for i in range(0,n_M):
    for j in range(0,len(db)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()

level = np.linspace(0,1,7)

cp=ax.contourf(db,m,h,levels = level)
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("M",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()