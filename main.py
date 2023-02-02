# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:43:15 2022

@author: aflatres
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
######################
#Impact of probability of establishment (P) and fecundity benefits b_y-b_x on level of stay-and-help
data = pd.read_csv('result_p_b.csv')

d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))
d_b = d_b[0:n_d_b]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/n_d_b)
b = np.linspace(1,8,n_d_b)
b=b[n_d_b-n_p:]

P = np.zeros(n_p)
for i in range(0,int(len(p)/n_d_b)):
    P[i] = p[i*len(d_b)]

h = np.zeros((n_p,n_d_b))
k = 0

for i in range(0,n_p):
    for j in range(0,len(d_b)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
hx=h
level = np.linspace(0,1,7)

cp=ax.contourf(d_b,b,h,levels=level)
level = np.linspace(0,0,1)
ax.contour(d_b,b,h,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
plt.rc('axes', labelsize=5)
####################
#Probability of establishment (b) and fecundity benefits
data = pd.read_csv('result_p_b.csv')

d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))
d_b = d_b[0:n_d_b]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])-np.array(data.loc[:,"h_y"])
h_data = h_data[p>0]
p=p[p>0]
n_p = int(len(p)/n_d_b)
b = np.linspace(1,8,n_d_b)
b=b[n_d_b-n_p:]

P = np.zeros(n_p)
for i in range(0,int(len(p)/n_d_b)):
    P[i] = p[i*n_d_b]

h = np.zeros((n_p,n_d_b))
k = 0

for i in range(0,n_p):
    for j in range(0,len(d_b)):
        h[i,j] = h_data[k]
        k+=1
fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
hx=h
level = np.linspace(0,0.2,7)

cp=ax.contourf(d_b,b,h,levels=level)
level = np.linspace(0,0,1)
ax.contour(d_b,b,h,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
plt.rc('axes', labelsize=5)
####################

#Probability of establishment (b) and fecundity benefits
data = pd.read_csv('result_p_b.csv')

d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))
d_b = d_b[0:n_d_b]

p = np.array(data.loc[:,"P"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]
hy_data = np.array(data.loc[:,"h_y"])
hy_data = hy_data[p>0]
p=p[p>0]
n_p = int(len(p)/n_d_b)
b = np.linspace(1,8,n_d_b)
b=b[n_d_b-n_p:]
s = np.linspace(0.1,0.99,n_d_b)
s=s[n_d_b-n_p:]
P = np.zeros(n_p)
for i in range(0,int(len(p)/n_d_b)):
    P[i] = p[i*len(d_b)]

h = np.zeros((n_p,n_d_b))

k = 0

for i in range(0,n_p):
    for j in range(0,n_d_b):
       
        h[i,j] = (b[i]*np.exp(-hy_data[k]*b[i])-(b[i]+d_b[j])*(np.exp(-hy_data[k]*(b[i]+d_b[j]))))
        k+=1
fig, ax=plt.subplots()


cp=ax.contourf(d_b,b,h,cmap="Greys")
level = np.linspace(0,0,1)
ax.contour(d_b,b,h,levels = level,colors = "orange")
ax.contour(d_b,b,hx,levels = level,colors = "red")

fig.colorbar(cp,label = "$T_x'-(1-T_y)'$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
##################

#Probability of establishment (b) and fecundity benefits
data = pd.read_csv('result_p_b_05.csv')

d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))

d_b = d_b[0:n_d_b]

b = np.linspace(1,8,n_d_b)
p = np.array(data.loc[:,"P"])

h_data = np.array(data.loc[:,"h_x"])
p=p[p>0]


n_b = len(b)
n_p = int(len(p)/n_d_b)

h = np.zeros((n_b,n_d_b))

k = 0

for i in range(0,n_b):
    for j in range(0,n_d_b):
        h[i,j] = h_data[k]
        k+=1
h = h[n_d_b-n_p:,:]
b=b[n_d_b-n_p:]
fig, ax=plt.subplots()

level = np.linspace(0,1,7)
cp=ax.contourf(d_b,b,h,levels = level)
level = np.linspace(0,0,1)

ax.contour(d_b,b,h,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
##################
# Same but with GA
data = pd.read_csv('result_p_b2.csv')

d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))
d_b = d_b[0:n_d_b]

p = np.array(data.loc[:,"P"])
p2 =np.array(data.loc[:,"P2"])
h_data = np.array(data.loc[:,"h_x"])
h_data = h_data[p>0]

p=p[p>0]
p2=p2[p2>0]
n_p = int(len(p)/len(d_b))
P = np.zeros(n_p)
P_2 = np.zeros(n_p)
for i in range(0,int(len(p)/n_d_b)):
    P[i] = p[i*n_d_b]
    P_2[i] = p2[i*n_d_b]
h = np.zeros((n_p,n_d_b))
k = 0

for i in range(0,n_p):
    for j in range(0,n_d_b):
        h[i,j] = h_data[k]
        k+=1

fig, ax=plt.subplots()
level = np.linspace(0,1,7)
b = np.linspace(1,8,n_d_b)
b=b[n_d_b-n_p:]
cp=ax.contourf(d_b,b,h,levels = level)
level = np.linspace(0,0,1)
ax.contour(d_b,b,h,levels = level,colors = "red")
fig.colorbar (cp,label = "$h_x$")
#Add a colorbar to a plot
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("$P$ of establishment",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)


######################
#Probability of establishment and hx
fig, ax=plt.subplots()
p = np.array(data.loc[:,"P"])
p2 = np.array(data.loc[:,"P2"])
h= np.array(data.loc[:,"h_x"])
b= np.array(data.loc[:,"d_b"])

n_p = 20


P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,int(len(p)/n_d_b)):
    P1[i] = p2[18+i*n_d_b]-p[18+i*n_d_b]
    P1[i] = P1[i]/p[18+i*n_d_b]
    H1[i] = h[18+i*n_d_b]
    P2[i] = p2[14+i*n_d_b]-p[14+i*n_d_b]
    P2[i] = P2[i]/p[14+i*n_d_b]
    H2[i] = h[14+i*n_d_b]
    P3[i] = p2[10+i*n_d_b]-p[10+i*n_d_b]
    P3[i] = P3[i]/p[10+i*n_d_b]
    H3[i] = h[10+i*n_d_b]
    

P1 = P1[H1>0]
H1 = H1[H1>0]
P2 = P2[H2>0]
H2 = H2[H2>0]
P3 = P3[H3>0]
H3 = H3[H3>0]

plt.scatter(H1,P1)
plt.scatter(H2,P2)
plt.scatter(H3,P3)

#plt.xlabel('$h_x$',fontsize = 20.0)
#plt.ylabel("$\Delta P/P$",fontsize = 20.0)
plt.legend(["$\Delta b=0.89$","$\Delta b=0.68$","$\Delta b=0.47$"],fontsize =13.0, bbox_to_anchor=[1.0,0.4])
#plt.title("Cooperation level $h_x$ value vs probability of establishment",fontsize = 15.0)
########################
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
for i in range(0,int(len(p)/n_d_b)):
    P1[i] = p[18+i*n_d_b]
    H1[i] = h[18+i*n_d_b]
    P2[i] = p[14+i*n_d_b]
    H2[i] = h[14+i*n_d_b]
    P3[i] = p[10+i*n_d_b]
    H3[i] = h[10+i*n_d_b]
P1 = P1[P1>0]
H1 = H1[H1>0]
P2 = P2[P2>0]
H2 = H2[H2>0]
P3 = P3[P3>0]
H3 = H3[H3>0]
plt.scatter(H1,P1)
plt.scatter(H2,P2)
plt.scatter(H3,P3)

#plt.xlabel('$h_x$',fontsize = 20.0)
#plt.ylabel("Probability of establishment",fontsize = 20.0)
plt.legend(["$\Delta b=0.89$","$\Delta b=0.68$","$\Delta b=0.47$"],fontsize =13.0)
#plt.title("Cooperation level $h_x$ value vs probability of establishment",fontsize = 15.0)


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
#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("Inbreeding coeff $\Phi$",fontsize = 15.0)
#plt.title("Contour plot of CSS $h_x$ vs $\Delta b$ and $\Phi$ ",fontsize =15.0)



##############################
#impact of phi and ga on reproductive value
data = pd.read_csv('result_phi_bx_ga2.csv')
d_b = np.array(data.loc[:,"d_b"])
n_d_b = int(np.sqrt(len(d_b)))
d_b = db[0:n_d_b]



phi = np.array(data.loc[:,"phi"])
h_data = np.array(data.loc[:,"GA"])


n_phi = int(len(phi)/n_d_b)
Phi = np.zeros(n_phi)
for i in range(0,n_phi):
    Phi[i] = phi[i*n_d_b]
nu = np.zeros((n_phi,n_d_b))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_d_b):
        nu[i,j] = h_data[k]
        k+=1


fig, ax = plt.subplots()

level = np.linspace(0,0.4,7)
cp=ax.contourf(d_b,Phi,nu,cmap = 'cividis')

fig.colorbar (cp)#Add a colorbar to a plot
level = np.linspace(0,0,1)
cp=ax.contour(d_b,Phi,h,levels= level, colors = "red")

#plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
#plt.ylabel("Inbreeding coeff $\Phi$",fontsize = 15.0)
#plt.title(" Effect of $\Phi$ on repro value ", fontsize = 15.0)

