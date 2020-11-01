#2019 By Kesson
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z


def mediator(y, t, mu1, mu2, k):
    u0, u1 = y
    dydt = [mu1*u0/1j + k*u1/1j,
            k*u0/1j + mu2*u1/1j]
    return dydt

def dndt(y0,t,mu1,mu2,k,size):
    y=[]
    y.append(y0)
    for i in range(size-1):
        z = odeintz(mediator, y[-1], [t[i],t[i+1]], args=(np.float(mu1[i]),np.float(mu2[i]),np.float(k[i])))
        y.append(list(z[-1]))

    y=np.array(y)
    psi1=y[:,0]
    psi2=y[:,1]

    dn11=1j*((mu1*np.conj(psi1)+k*np.conj(psi2))*psi1-np.conj(psi1)*(mu1*psi1+k*psi2))
    n1=psi1.real**2+psi1.imag**2
    dn22=1j*((mu2*np.conj(psi2)+k*np.conj(psi1))*psi2-np.conj(psi2)*(mu2*psi2+k*psi1))
    n2=psi2.real**2+psi2.imag**2
    return dn11,dn22,n1,n2

def mud(mu2,size,dmu,pmu,x,y,w):
    pre=int(size*x)
    post=int(size*y)
    w=int(size*w)
    i=0
    while 1:
        if pre<i<post:
            mu2[i]= dmu*pmu
            i=i+w
        elif i>post:
            break
        else:
            i=i+1
            
    return mu2
#Coefficient
y0=[0.24,0.76]
size=1000
t = np.linspace(0, 1000, size) #dt too small not good
dmu =0.01
mu1 = np.ones([size])*dmu
mu2 = np.ones([size])*dmu
k = 0.04*np.ones([size])
pmu = 500
nu = 0.01
nu1 = 0.03
nu2 = 0.02
NAME = 'Memory16'
mu2=mud(mu2,size,dmu,pmu,0.1,0.15,nu)
mu2=mud(mu2,size,dmu,pmu,0.17,0.28,nu1)
mu2=mud(mu2,size,dmu,pmu,0.32,0.44,nu2)
mu2=mud(mu2,size,dmu,pmu,0.5,1,0.03)
muu=mu2-mu1
# solve ODES
dn11,dn22,n1,n2=dndt(y0,t,mu1,mu2,k,size)

# plot results
#u0 = y[:, 0].real ** 2+y[:, 0].imag ** 2 # + \
#  z[:, 2].real ** 2 + z[:, 0].imag ** 2 + \
#  z[:, 1].imag ** 2 + z[:, 2].imag ** 2
#u1 = y[:, 1].real ** 2+y[:, 1].imag ** 2
a=plt.figure()
#plt.plot(t, u0, c='black', lw='3', label='u0')
n = (n1+n2)
dnn = dn11+dn22
dn=dn11-dn22
#plt.plot(t, dn, c='black', lw='3', label='u1')
plt.plot(t, dn, c='black', lw='3', label='u1')
plt.plot(t,k)
plt.plot(t,-k)
#plt.plot(t, dnn[0,:], c='black', lw='3', label='u1')
#plt.plot(t, u2, c='red', lw='3', label='u2')
#plt.plot(t, u0+u1+u2, c='blue', lw='3', label='sum')
#plt.legend()
plt.savefig('aa/nonspike'+NAME+'.png')
a=plt.figure()
plt.plot(muu)
plt.savefig('aa/nonspikepeak'+NAME+'.png')
