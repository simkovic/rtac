## The MIT License (MIT)
##
## Copyright (c) <2016> <Matus Simkovic>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

import numpy as np
import pylab as plt
from misc import *
from matustools.matusplotlib import subplotAnnotate
from scipy.stats import norm

__all__=['contourPcaxes','transformation']

def contourPcaxes(mu,sd,r,N=200,maxRT=15):
    X,Y=np.meshgrid(np.linspace(0,maxRT,N),np.linspace(0,1,N))
    dat=[X,Y]
    dat=np.concatenate([dat[0].flatten()[:,np.newaxis],
            dat[1].flatten()[:,np.newaxis]],1)
    lat=data2latent(dat)
    pc=latent2pc(lat-np.atleast_2d(mu),r=r,sd=sd)
    prob=norm.pdf(np.linalg.norm(pc,axis=1))
    prob=np.reshape(prob,[N,N])
    prob[np.isnan(prob)]=0
    plt.contour(X,Y,prob,np.linspace(0,0.5,11)[1:],colors='k')
    plt.xlim([np.min(X),np.max(X)])
    plt.ylim([np.min(Y),np.max(Y)])
    plt.grid(False) 

def transformation(points,mu,sd,r,detailed=False,pmrkr='.k',cmrkr='r-',cscale=1,
                    clw=2,csmpl=16,limRT=[0,15],limHR=[0,1]):
    '''
        points - Nx2 array with N subject/item estimates of mean log response 
            time and mean logit hit rate
        mu - array with two elements corresponding to population MEAN of the 
            log response time and logit hit rate respectively
        sd - array with two elements corresponding to STANDARD DEVIATION of the 
            log response time and logit hit rate respectively
        r - CORRELATION between log response time and logit hit rate
        detailed - if True will show 6 rather than 3 cells
        pmrkr -  marker/color specification of points for matplotlib.plot
        cmrkr - marker/color specification for pc axes
        cscale - length of the principal component axes, 
                cscale=1 -> 1 standard deviation
        clw - line width of the principal component axes
        csmpl -  nr of points making line of a pc axis, 
                higher number -> higher resolution
        limRT - response time axis limits (x axis in A)
        limHR - hit rate axis limits(y axis in A)
    '''
    assert points.ndim==2 and points.shape[1]==2
    # pars
    pLT=points
    N=points.shape[0] 
    lmb= np.diag([np.sqrt(1+r),np.sqrt(1-r)]);
    lmbI=np.diag([1/np.sqrt(1+r),1/np.sqrt(1-r)])
    U=np.array([[np.cos(np.pi/4),np.sin(np.pi/4)],
        [-np.sin(np.pi/4),np.cos(np.pi/4)]]);UI=U.T
    H=np.diag(sd);HI=np.diag(1/np.float32(sd))
    temp=np.linspace(-cscale,cscale,csmpl)
    # compute pc axes and points in each space
    c1UG=np.array([temp,np.zeros(temp.shape)]).T
    c2UG=np.array([np.zeros(temp.shape),np.copy(temp)]).T
    c1SC=c1UG.dot(lmb);c2SC=c2UG.dot(lmb)
    c1R4=c1SC.dot(U);c2R4=c2SC.dot(U)
    c1DM=c1R4.dot(H);c2DM=c2R4.dot(H)
    c1LT=c1DM+np.atleast_2d(mu);c2LT=c2DM+np.atleast_2d(mu)
    c1DT=latent2data(c1LT);c2DT=latent2data(c2LT)
    c1=[c1DT,c1LT,c1DM,c1R4,c1SC,c1UG];c2=[c2DT,c2LT,c2DM,c2R4,c2SC,c2UG]
    pDM=pLT-np.atleast_2d(mu);pDT=latent2data(pLT);pDM=pLT-np.atleast_2d(mu)
    pR4=pDM.dot(HI);pSC=pR4.dot(UI);pUG=pSC.dot(lmbI)
    p=[pDT,pLT,pDM,pR4,pSC,pUG]
    # create plot
    axlim=[limRT+limHR,[-3*sd[0]+mu[0],3*sd[0]+mu[0],-3*sd[1]+mu[1],
            3*sd[1]+mu[1]],[-3*sd[0],3*sd[0],-3*sd[1],3*sd[1]],[-3,3,-3,3],
           [-3*lmb[0,0],3*lmb[0,0],-3*lmb[1,1],3*lmb[1,1]],[-3,3,-3,3]]
    axaspect=[None,sd[0]/sd[1],sd[0]/sd[1],1,1,1]
    axlabel=[['Response time','Hit rate'],['Log response time','Logit hit rate'],
            None,None, None,['Speed-accuracy trade-off','Ability-difficulty']]
    if detailed: indx=range(6)
    else: indx=[0,1,5]
    for j in range(len(indx)): 
        i=indx[j]
        plt.subplot(1,len(indx),j+1)
        plt.plot(c1[i][:,0],c1[i][:,1],cmrkr,lw=clw)
        plt.plot(c2[i][:,0],c2[i][:,1],cmrkr,lw=clw)
        plt.plot(p[i][:,0],p[i][:,1],pmrkr)
        plt.xlim(axlim[i][:2]);plt.ylim(axlim[i][2:]);
        if not axaspect[i] is None: plt.gca().set_aspect(axaspect[i])
        if not axlabel[i] is None: 
            plt.xlabel(axlabel[i][0])
            plt.ylabel(axlabel[i][1])
        subplotAnnotate(loc='se')
