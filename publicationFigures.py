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
from scipy.optimize import fmin
from matustools.matusplotlib import *

mu=[2,3];sd=[1/3.,2/3.];r=0.5

def syntheticExample():
    np.random.seed(5)
    pts=np.random.randn(200,2)
    pts=pc2latent(pts,sd=sd,r=r)+np.atleast_2d(mu)
    plt.figure(figsize=(20,6))
    transformation(pts,mu=mu,sd=sd,r=r,detailed=True,cscale=2,
        limRT=[0,16],limHR=[0.8,1])
        
def sdContours():
    for i in range(5):
        for j in range(5):
            sdrt=np.linspace(0,2,6)[i+1]
            sdac=np.linspace(0,4,6)[j+1]
            subplot(5,5,i*5+j+1)
            contourPcaxes(mu=[1,1],sd=[sdrt,sdac],r=0.5,maxRT=16)
            c1,c2=pcAxes(mu=[1,1],sd=[sdrt,sdac],r=0.5,scale=2)
            plt.plot(c1[:,0],c1[:,1],color='r',lw=1)
            plt.plot(c2[:,0],c2[:,1],color='r',lw=1)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            
def muContours():
    for i in range(5):
        for j in range(5):
            murt=np.linspace(0,1,6)[i+1]
            muac=np.linspace(-2,2,6)[j+1]
            subplot(5,5,i*5+j+1)
            contourPcaxes(mu=[murt,muac],sd=[1.,2.],r=0.5,maxRT=15)
            c1,c2=pcAxes(mu=[murt,muac],sd=[1.,2.],r=0.5,scale=2)
            plt.plot(c1[:,0],c1[:,1],color='r',lw=1)
            plt.plot(c2[:,0],c2[:,1],color='r',lw=1)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([]) 
            
def rContours():
    for i in range(5):
        for j in range(3):
            r=np.linspace(-0.9,0.9,5)[i]
            muac=[1,0,-1][j]
            subplot(3,5,j*5+i+1)
            contourPcaxes(mu=[0.5,muac],sd=[1.,2.],r=r,maxRT=15)
            c1,c2=pcAxes(mu=[0.5,muac],sd=[1.,2.],r=r,scale=2)
            plt.plot(c1[:,0],c1[:,1],color='r',lw=1)
            plt.plot(c2[:,0],c2[:,1],color='r',lw=1)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])          
    

def logisticApproximation():
    x=np.linspace(-5,5,501)
    y1=1/(1+np.exp(-x))
    subplot(1,2,1)
    plt.plot(x,y1,'y',lw=5)
    def errfun(th):
        xerf=np.linspace(0.3,4,501)
        yerf=1/(1+np.exp(-xerf))
        return np.square(1-np.exp(-xerf*th[0]+th[1])-yerf).mean()
    th=fmin(errfun,[0.7,-0.5])
    print th
    th=[0.782,-0.566]
    y3=1-np.exp(-x*th[0]+th[1])
    plt.plot(x,y3,'r')
    y4=x/4.+0.5
    plt.plot(x,y4,'g')
    y5=np.exp(x*th[0]+th[1])
    plt.plot(x,y5,'b')
    plt.ylim([0,1])
    plt.xlabel('Logit hit rate')
    plt.ylabel('Hit rate')
    plt.legend(['Logistic','1-Exponential','Linear','Exponential'],loc=4)
    subplotAnnotate(loc='nw')
    subplot(1,2,2)
    plt.grid(True,axis='x')
    plt.plot(y1,np.abs(y3-y1),'r');plt.ylim([-0,0.02]);
    plt.plot(y1,np.abs(y4-y1),'g')
    plt.plot(y1,np.abs(y5-y1),'b')
    plt.gca().set_xticks(np.linspace(0,1,11));
    plt.xlabel('Hit rate')
    plt.ylabel('Absolute error')
    subplotAnnotate(loc='nw')
    
if __name__ == '__main__':
    import os
    DPI=300
    figpath=os.getcwd()+os.path.sep+'figures'+os.path.sep
    
    plt.figure()
    syntheticExample()
    plt.savefig(figpath+'syntheticExample',dpi=DPI)
    
    plt.figure(figsize=(20,20))
    sdContours()
    plt.savefig(figpath+'sdContours',dpi=DPI)
    
    plt.figure(figsize=(20,20))
    muContours()
    plt.savefig(figpath+'muContours',dpi=DPI)
    
    plt.figure(figsize=(20,10))
    rContours()
    plt.savefig(figpath+'rContours',dpi=DPI)
    
    plt.figure(figsize=(10,5))
    logisticApproximation()
    plt.savefig(figpath+'logisticApproximation',dpi=DPI)
    
    
