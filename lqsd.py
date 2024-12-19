# code to simulate linear quantum state diffusion

# according to
# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.54.2664


# the main equation of this evolution is 
# d psi =  -iH    psi dt + ∑ L  psi dW    with dW ~ W    - W
#             eff    t     m  m       t          t   t+dt   t

# This has some kind of unnormalized solution like https://www.youtube.com/watch?v=_lRVj0vSW9Q&t=1134s
# geonmetric brownian motion in ito, in strato i think its the plain exponential, but  
#                      +
# psi0 exp ( (-iH   - L L/2) t  + L dW )
#                eff                  t

# So in principle it is possible to trotterize and apply the usual machinerty of 
# matrix product states (MPS) and get larger systems.

import jax 
from functools import partial
import numpy
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, Euler, Heun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree, Dopri5
import numpy as np
from jax.scipy.linalg import expm
import h5py
COMPLEX_TYPE  = jax.numpy.complex64


sX = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)
sY = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX_TYPE)
sZ = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)
sM = np.array([[0, 1.], [0, 0.]], dtype=COMPLEX_TYPE) 
XX = np.kron(sX, sX)
YY = np.kron(sY, sY)
ZZ = np.kron(sZ, sZ)
X1 = np.kron(sX, np.eye(2))
sN = sM.T@sM
XN = np.kron(sN, sX)
NX = np.kron(sX, sN)
X2 = np.kron(np.eye(2), sX)
jXN, jNX, jN, jM = map(jnp.array, [ XN ,NX, sN,sM])

def nk(key):
    key,subkey = jr.split(key)
    return subkey


def ham(params, qnum):
    '''
    hamiltonian 
    '''
    h = params
    out = np.zeros((2**qnum, 2**qnum), dtype=COMPLEX_TYPE)
    for i in range(qnum-1):
        out +=  h * np.kron(
           np.eye(2**i,dtype=COMPLEX_TYPE),
           np.kron(jNX, np.eye(2**(qnum-i-2),dtype=COMPLEX_TYPE)))
    for i in range(qnum-1):
        out +=  h * np.kron(
           np.eye(2**i,dtype=COMPLEX_TYPE),
           np.kron(jXN, np.eye(2**(qnum-i-2),dtype=COMPLEX_TYPE)))
    return jnp.array(out)


def heff(params, qnum):
    '''
    Effective hamiltonian 
    '''
    h = params
    out = np.zeros((2**qnum, 2**qnum), dtype=COMPLEX_TYPE)
    for i in range(qnum-1):
        out +=  h * np.kron(
           np.eye(2**i,dtype=COMPLEX_TYPE),
           np.kron(jNX, np.eye(2**(qnum-i-2),dtype=COMPLEX_TYPE)))
    for i in range(qnum-1):
        out +=  h * np.kron(
           np.eye(2**i,dtype=COMPLEX_TYPE),
           np.kron(jXN, np.eye(2**(qnum-i-2),dtype=COMPLEX_TYPE)))
    for i in range(qnum):
        out +=  -1j *0.5*  np.kron(
            np.eye(2**i,dtype=COMPLEX_TYPE),
            np.kron(sM.T.conj() @ sM, np.eye(2**(qnum-i-1),dtype=COMPLEX_TYPE)))
    return jnp.array(out)

def dissipator( qnum,O=sM):
    '''
    quantum jumps 
    '''
    out = np.zeros((2**qnum, 2**qnum), dtype=COMPLEX_TYPE)
    #jumps = jnp.array([jnp.kron(jnp.eye(2**i,dtype=COMPLEX_TYPE),jnp.kron(O, jnp.eye(2**(qnum-i-1), dtype=COMPLEX_TYPE))) for i in range(qnum)])
    jumps = jnp.array([ jnp.kron(
             np.eye(2**i),
             np.kron( sM, np.eye(2**(qnum-i-1)))) for i in range(qnum)  ] )
    return jumps

def basis_state(spins):
    out = np.array([1-spins[0],spins[0]])
    for s in spins[1:]:
        spin = np.array([s,1-s])
        out = np.kron(out,spin)
    return jnp.array(out,dtype=COMPLEX_TYPE)

def exact_solveLQSD(Heff,jumps,qnum,params,t0,t1,psi0,dt,key,Oi,index=0):
    nsteps = int((t1-t0)/dt)
    ts = jnp.linspace(t0,t1,nsteps)
    #xis = jnp.sqrt(0.5*dt) * (jr.normal(key,shape=(nsteps,qnum)) + 1j * jr.normal(nk(key),shape=(nsteps,qnum)))
    def _oit(psi,key):
        xi = jnp.sqrt(0.5*dt) * (jr.normal(key,shape=(qnum,)) + 1j * jr.normal(nk(key),shape=(qnum,)))
        _E = jnp.einsum('i,mji,j->m',psi.conj(),jumps.conj(),psi)
        _EE = -1j * dt* jnp.einsum('mij,m->ij',jumps,_E)
        G0 = expm(-1j * Heff * dt * 0.5 +  _EE * 0.5  )
        psi =  G0 @ expm(jnp.einsum('ijk,i->jk', jumps, xi)) @ G0 @ psi
        #psi = psi / jnp.sqrt(psi.T.conj()@psi)
        oit =  psi.T.conj() @ (Oi @ psi) #/ psi.T.conj()@psi
        return psi,oit.real
    keys = jr.split(key,num = nsteps)
    sol,ys = jax.lax.scan(f=_oit, init=psi0,xs=keys)
    return ys, ts 

def lindblad(Heff,jumps,rho):
    com = - 1j * (Heff @ rho - rho @ Heff.conj().T)
    ErhoE = jnp.einsum('mil,lk,mjk->ij',jumps, rho, jumps.conj())
    return com + ErhoE 


def solveLindblad(H,jumps,qnum,params,t0,t1,rho0,dt):
    nsteps = int((t1-t0)/dt)
    drift = lambda t, rho, args: lindblad(H,jumps,rho) 
    term = ODETerm(drift)
    solver = Dopri5() 
    saveat = SaveAt(ts=jnp.linspace(t0, t1, nsteps))
    sol = diffeqsolve(term, solver, t0, t1, dt0=dt, y0=rho0, saveat=saveat)
    return sol

#@jax.jit
def create_dataset(yss,key):
    for i in range(len(yss)):
        key=nk(key)
        ys,ts=exact_solveLQSD(Heff,jumps,qnum,params,t0,t1,psi0,dt,key,Oi)
        yss = yss.at[i].set(ys)
    return yss, ts

if __name__ == '__main__':
    t0, t1 = 0.,4. 
    key =jr.PRNGKey(numpy.random.randint(2**14))
    qnum = 4
    params = 3. 
    dt = 0.01
    n_steps = int((t1-t0)/dt)
    n_samples = 40
    psi0  =  jnp.zeros(2**qnum,dtype=COMPLEX_TYPE)  #basis_state([0 for _ in range(num_sites)])
    psi0  =  psi0.at[-1].set(1.)#(2**num_sites,dtype+COMPLEX_TYPE)  #basis_state([0 for _ in range(num_sites)])
    rho0 =  jnp.einsum('i,j-> ij',psi0.conj(),psi0 )#basis_state([1,1,1,1])
    Heff = heff(params,qnum)
    jumps = dissipator(qnum)
    nsteps = int((t1-t0)/dt)
    index = int( 0.5 * qnum  ) 
    O = jN
    Oi = jnp.kron(
            jnp.eye(2**index),
            jnp.kron(O, jnp.eye(2**(qnum-index-1))))

    #key=jr.PRNGKey(0)
    yss=[]
    yss=jnp.empty((n_samples,n_steps))
    yss, ts = create_dataset(yss,key) 
    #h5f = h5py.File('./dataset/data_cp_h_{3.}.h5', 'w')
    #h5f.create_dataset('dataset_1', data=np.array(yss))


    ym = jnp.array(yss).mean(axis=0)
    plt.plot(ts, ym, color='tab:red')
    solind = solveLindblad(Heff,jumps,qnum,params,t0,t1,rho0,dt)
    tr = jnp.einsum( 'tii->t',solind.ys  )#   /jnp.einsum('ti,ti->t', sol.ys,sol.ys.conj()  ).real
    nt = jnp.einsum( 'ij,tji->t',Oi,solind.ys  )#/tr#   /jnp.einsum('ti,ti->t', sol.ys,sol.ys.conj()  ).real
    plt.plot(solind.ts, nt, color='tab:orange',label = 'exact',linestyle = 'dashed')
    plt.plot(solind.ts, tr, color='tab:purple',label = 'trace')
    plt.xlabel('t')
    plt.ylabel('n1')

    plt.legend()
    plt.show()
    
