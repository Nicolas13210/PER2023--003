from random import randint, sample, randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from math import tan, atan
import os



def lisse_im_fst(im,pas_resolution):
    m=[]
    for i in range(len(im)//pas_resolution):
        lgn=[]
        for j in range(len(im[0])//pas_resolution):
            L=[im[i*pas_resolution][j*pas_resolution]/256]
            lgn.append(moy_color(L))
        m.append(lgn)
    return m


def moy_color(L):
    m0,m1,m2=0,0,0
    for e in L:
        r,v,b=e
        m0=m0+r
        m1=m1+v
        m2=m2+b
    m=(m0+m1+m2)/len(L)/3
    return m

palette=[0.3,1] # le premier elt de palette doit correspondre au noir (le tapis)

def cproche(c):
    def ecart(c1,c2):
        return abs(c1-c2)
    cp0=palette[0]
    cp1=palette[1]
    if ecart(c,cp0)<ecart(c,cp1):
        return cp0
    else:
        return cp1

def traite_im(im):
    r=[]
    for i in range(len(im)):
        lgn=[]
        for j in range(len(im[0])):
            cp=cproche(im[i][j])
            lgn.append(cp)
        r.append(lgn)
    return r

def remplissage(im,i,j,col_cible,col_rempl):
    if im[i][j]==col_cible:
        im[i][j]=col_rempl
        if i>0:
            remplissage(im,i-1,j,col_cible,col_rempl)
        if i<(len(im)-1):
            remplissage(im,i+1,j,col_cible,col_rempl)
        if j>0:
            remplissage(im,i,j-1,col_cible,col_rempl)
        if j<(len(im[0])-1):
            remplissage(im,i,j+1,col_cible,col_rempl)


def find_white(im,i_s,j_s):
    for i in range(i_s,len(im)):
        for j in range(len(im[0])):
            if im[i][j]==1:
                return i,j
    return 'none'

def groupe(im):
    fw=True
    i_s,j_s=0,0
    while fw:
        fwh=find_white(im,i_s,j_s)
        if fwh=='none':
            fw=False
        else:
            i,j=fwh
            col=uniform(0.4,0.8)
            remplissage(im,i,j,1,col)
            i_s,j_s=i,j

def liste_blancs(imb):
    L=[]
    for i in range(len(imb)):
        for j in range(len(imb[0])):
            if imb[i][j]!=palette[0]:
                L.append([i,j])
    return L


def vecteurs(imt,n):
    vects=[]
    LB=liste_blancs(imt)
    while len(vects)<n:
        S1=sample(LB,1)[0]
        S2=sample(LB,1)[0]
        if S1[0]<S2[0]:
            i1,j1=S1
            i2,j2=S2
        else:
            i1,j1=S2
            i2,j2=S1
        if imt[i1][j1]==imt[i2][j2] and imt[i1][j1]!=palette[0]:
            v=[i1-i2,j1-j2]
            vects.append(v)
    return vects


def vecteur_moyen(imt,n):
    vects=vecteurs(imt,n)
    s_i,s_j=0,0
    for v in vects:
        s_i=s_i+v[0]
        s_j=s_j+v[1]
    return [s_i/n,s_j/n]


def disp_vm(im,vm):
    mid=len(im[0])//2-1
    bot=len(im)-1
    N=128
    for k in range(N):
        im[int(bot+3*k*vm[0]/N)][int(mid+3*k*vm[1]/N)]=1

def action(im,v,vitesse,rotation):
    #vitesse=4
    #rotation=0.1
    vr=[0,0]
    vr[0]=v[0]/len(im)
    vr[1]=v[1]/len(im)
    #print(vr)
    angle=atan(vr[1]/vr[0])
    a=[vitesse*(abs(vr[0])+(1-angle/1.6)/2),angle*rotation]
    return a

def action2(im,v,v_anc,vitesse,rotation):
    #vitesse=4
    #rotation=0.1
    p=1
    vr=[0,0]
    vr[0]=(v[0]+p*v_anc[0])/len(im)/(1+p)
    vr[1]=(v[1]+p*v_anc[1])/len(im)/(1+p)
    #print(vr)
    angle=atan(vr[1]/vr[0])
    a=[vitesse*(abs(vr[0])+(1-angle/1.6)/2),angle*rotation]
    return a


def comp(im,pas,nv):
    #print(type(im))
    image_vecteur = np.frombuffer(im.data, dtype="uint8")
    image_matrice = image_vecteur.reshape(240,424,3)
    image_matrice=image_matrice[100:]
    #t0=time.clock()
    iml=lisse_im_fst(image_matrice,pas)
    imt=traite_im(iml)
    groupe(imt)
    v=vecteur_moyen(imt,nv)
    print(v)
    #t4=time.clock()
    #print(t4-t0)
    disp_vm(imt,v)
    a=action(imt,v,2,0.1)
    # [vitesse m/s ; angle rotation (sens trigo)]
    print('action : ')
    print(a)
    print('')
    return a



          
