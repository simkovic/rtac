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
__all__=['data2latent','latent2data','latent2percentGain','pc2latent',
         'latent2pc', 'pcAxes']
def data2latent(data):
    res=np.zeros(data.shape)
    res[:,0]=np.log(data[:,0])
    res[:,1]=np.log(data[:,1])-np.log(1-data[:,1])
    return res
    
def latent2data(latent):
    '''
        latent - Nx2 array with log response time in the first column
                 and logit hit rate in the second column
        returns Nx2 array with exp of latent[:,0] in first column 
                and logit of latent[:,1] in the second
    '''
    latent=np.array(latent,ndmin=2)
    res=np.zeros(latent.shape)
    res[:,0]=np.exp(latent[:,0])
    res[:,1]=1/(1+np.exp(-latent[:,1]))
    return res
    
def logisticAppr(delta,th=0.782): return np.exp(th*delta)

def latent2percentGain(latent,G0,decim=-1):
    latent=np.array(latent,ndmin=2)
    res=np.zeros(latent.shape)
    res[:,0]=np.exp(latent[:,0])
    res[:,1]=logisticAppr((2*G0-1)*latent[:,1])
    res=(res-1)*100
    if decim<0: return res
    else: return np.round(res,decimals=decim)

def pc2latent(pc,sd,r):
    pc=np.array(pc,ndmin=2,dtype=np.float32);latent=np.zeros(pc.shape)
    sd=np.array(sd,ndmin=2);c=np.cos(np.pi/4)
    pc[:,0]*=np.sqrt(1+r);pc[:,1]*=np.sqrt(1-r);
    latent[:,0]=c*(pc[:,0]-pc[:,1])*sd[:,0]
    latent[:,1]=c*(pc[:,0]+pc[:,1])*sd[:,1]
    return latent

def latent2pc(latent,sd,r):
    latent=np.array(latent,ndmin=2,dtype=np.float32);pc=np.zeros(latent.shape)
    sd=np.array(sd,ndmin=2,dtype=np.float32);c=np.cos(np.pi/4)
    latent/=sd
    pc[:,0]=c*(latent[:,0]+latent[:,1])/np.sqrt(1+r)
    pc[:,1]=c*(-latent[:,0]+latent[:,1])/np.sqrt(1-r)
    return pc
    
def pcAxes(mu,sd,r,csmpl=100,scale=1):
    # transformed pars
    temp=np.linspace(-scale,scale,csmpl)
    c1=np.array([temp,np.zeros(temp.shape)]).T
    c2=np.array([np.zeros(temp.shape),np.copy(temp)]).T
    c1=pc2latent(c1,sd,r)+np.atleast_2d(mu)
    c2=pc2latent(c2,sd,r)+np.atleast_2d(mu)
    c1=latent2data(c1)
    c2=latent2data(c2)
    return c1,c2
