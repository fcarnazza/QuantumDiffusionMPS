# This code is adapted from https://github.com/frederikwilde/differentiable-tebd
# by Paco Carnazza
# All errors reserved 

import jax 
import numpy
import os
from jax import checkpoint
from jax import jit
from functools import partial
import numpy as np
import jax.numpy as jnp
from svd import svd
import jax.random as jr
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
from lqsd import *

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
N1 = np.kron(sN,np.eye(2))
N2 = np.kron(np.eye(2),sN)
jXN, jNX, jN, jM,jN1,jN2,jZ = map(jnp.array, [ XN ,NX, sN,sM,N1,N2,sZ])

def nk(key):
    key,subkey = jr.split(key)
    return subkey
 
def apply_gate_totuple(tensor_tuple, gate):
    t1, t2 = tensor_tuple
    n1, n2, err_sqr = contract_and_split(t1, t2, gate)
    return jnp.stack([n1, n2]), err_sqr

 
def expmh(coeff, hmat):
    '''
    Matrix exponential of a Hermitian matrix multiplied by a coefficient.

    Args:
        coeff: float or complex.
        hmat: Hermitian matrix.

    Returns:
        array: matrix exponential of coeff * hmat.
    '''
    e, u = jnp.linalg.eigh(hmat)
    return u.dot(jnp.diag(jnp.exp(coeff * e))).dot(u.transpose().conj())
 
def mps_zero_state(num_sites, chi, rng=None, d=2):
    '''
    The all zero state: |00000...>

    Args:
        num_sites (int): Number of sites.
        chi (int): Bond dimension.
        rng (np.random.Generator): A RNG created with np.random.default_rng
            Default is None, in which case a fresh PRNG is created.
        d (int): Physical dimension.

    Returns:
        array: MPS
    '''
    mps = jnp.zeros((num_sites, chi, d, chi), dtype=COMPLEX_TYPE)
    for i in range(num_sites):
        mps = mps.at[i, 0, 1, 0].set(1.)
    return mps

def contract_and_split(n1, n2, gate):
    chi = n1.shape[2]
    d = n1.shape[1]
    n = jnp.tensordot(n1, n2, axes=(2, 0))
    c = jnp.tensordot(n, gate, axes=((1,2), (2,3))).transpose((0, 2, 3, 1))
    u, s, v = svd(c.reshape(d * chi, d * chi))
    # FIXME: derivative at zero
    s_sqrt = jnp.sqrt(s[:chi])
    truncation_error_squared = jnp.sum(s[chi:] ** 2)
    u_out = (s_sqrt * u[:, :chi]).reshape(chi, d, chi)
    v_out = (s_sqrt * v[:chi, :].transpose()).transpose().reshape(chi, d, chi)
    return u_out, v_out, truncation_error_squared


def apply_gate(mps, i, gate):
    n1, n2, err_sqr = contract_and_split(mps[i], mps[i+1], gate)
    mps = mps.at[i:i+2].set(jnp.array([n1, n2]))
    return mps, err_sqr

def norm_squared(mps):
    '''
    The squared norm of an MPS.
    '''
    def _update_left(left, tensor):
        '''For scan function.'''
        t1 = jnp.tensordot(left, tensor.conj(), axes=(1, 0))
        return jnp.tensordot(tensor, t1, axes=((0,1), (0,1))), None

    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = left.at[0, 0].set(1.)
    left, _ = jax.lax.scan(_update_left, left, mps)
    return left[0,0].real

def norm(mps):
    def _update_left(left, tensor):
        '''For scan function.'''
        return jnp.einsum('ij,isq,jsp->qp',left,tensor, tensor.conj()),None
    left = jnp.eye(mps.shape[1],  dtype=COMPLEX_TYPE)
    left, _ = jax.lax.scan(_update_left, left, mps)
    return jnp.sqrt(jnp.trace(left))
 
def one_siteE(mps,Op,site):
    '''
    Expectation value of operator Op at site.
    ONLY WORKS FOR N...
    '''
    tensorOpi =  jnp.copy(mps) 
    tensorOpi = tensorOpi.at[site].set( jnp.einsum('ij,pjq->piq', Op,tensorOpi[site])  ) 
    psiopsi = jnp.array([[mps[i],tensorOpi[i]] for i in range(len(mps))])

    def _update_left(left, tensor):
        '''For scan function.'''
        t1 = jnp.tensordot(left, tensor.conj()[0], axes=(1, 0))
        return jnp.tensordot( tensor[1], t1, axes=((0,1), (0,1))), None

    left = jnp.zeros((mps.shape[1], mps.shape[1]), dtype=COMPLEX_TYPE)
    left = left.at[0, 0].set(1.)
    left, _ = jax.lax.scan(_update_left, left, psiopsi  )
    return left[0,0].real #* jnp.sqrt(norm_squared(mps))


@partial(jit, static_argnums=[2])
def cp_mps_evolution(params, deltat, steps, mps,site,key):
    '''
    Args:
        params (array): Hamiltonian parameters.
        deltat (float): Trotter step size.
        steps (int): Number of Trotter steps.
        mps (array): Initial mps with shape (num_sites, chi, 2, chi).
    Returns:
        Array: output MPS after steps
        float: Cumulated errors
    '''
    h = params
    gate_left =   expm(-1.j * deltat * (h*jXN + h*jNX )).reshape(2,2,2,2)
    gate_middle = expm(-1.j * deltat * (h*jXN + h*jNX )).reshape(2,2,2,2)
    gate_right =  expm(-1.j * deltat * (h*jXN + h*jNX )).reshape(2,2,2,2)
    keys = jr.split(key,num = steps)
    mps, opi = jax.lax.scan(
            lambda m,x: trotter_step(m,x, gate_left, gate_middle, gate_right,deltat,site=site),
            mps,
            keys,
            length=steps
        ) 
    return mps, opi

def apply_gate_batched(mps, i_from, i_to, gate):
    '''
    Applies gate or gates to the MPS within a specified index range.
    
    Args:
        mps (array): MPS
        i_from (int): Index of the first MPS tensor to apply the gate to.
        i_to (int): Index of the first MPS tensor not to apply the gate to.
        gate (array): Either a 4-dimensional array, which is applied to all
            specified MPS tensors, or a 5-dimensional array with the first
            dimension being equal to the half the number of MPS tensors specified
            by [i_from: i_to].
    
    Returns:
        array: The MPS after contractions.
        float: The summed squares of the truncation errors.
    '''
    if mps.shape[0] % 2 != 0:
        raise ValueError('The number of sites must be even.')
    K = (i_to - i_from)
    if K % 2 != 0:
        raise ValueError('[i_from: i_to] must describe a slice of even length.')

    if len(gate.shape) == 4:
        apply = jax.vmap(apply_gate_totuple, in_axes=(0, None))
    elif len(gate.shape) == 5:
        apply = jax.vmap(apply_gate_totuple, in_axes=(0, 0))
    else:
        raise ValueError(f'gate has invalid shape. Should be either 4, or 5 dimensional, shape was {gate.shape}.')

    tensors, errs_sqr = apply(
        mps[i_from:i_to].reshape(K//2, 2, *mps.shape[1:]),
        gate
    )
    mps = mps.at[i_from:i_to].set(tensors.reshape(K, *mps.shape[1:]))
    return mps, jnp.sum(errs_sqr)



@checkpoint
def trotter_step(mps,key, gate_left, gate_middle, gate_right,deltat,site=0,Op=jN):
    '''
    Number of qubits must be even!
    Applies one Trotter step to mps.
    gate1 is applied to all but the last pair of qubits.
    gate2 is only applied to the last pair as displayed below.
    x represtents the inserion of the complex gaussian noise term 

    ```
    |  |  |  |  |  |  |  |
    [gl]  [gm]  [gm]  [gr]
    |  [gm]  [gm]  [gm]  |
    x  x  x  x  x  x  x  x  
    |  |  |  |  |  |  |  |
    ```

    Args:
        mps (array): Input MPS.
        gate_left (array): is applied to the first and second qubit.
            Involves one time step deltat.
        gate_middle (array): is applied to all other qubits.
            Involves one time step deltat. Can be a list of gates.
        gate_right (array): is applied to the second to last and last qubit.
            Involves one time step deltat.

    Returns:
        array: New MPS.
        float: Summed squared truncation errors.
    '''
    #mps =lnormalize(mps)
    norm_ = norm(mps)
    L = mps.shape[0]
    if L % 2 != 0:
        raise ValueError('The number of sites must be even.')
    trunc_err_sqr = 0.
    jps = lambda i: one_siteE(mps,jM.T,i)
    es = jax.lax.map(jps,jnp.array([ idx for idx in range(L)]))

    # even layer
    mps, err_sqr = apply_gate(mps, 0, gate_left)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate_batched(mps, 2, L-2, gate_middle)
    trunc_err_sqr += err_sqr
    mps, err_sqr = apply_gate(mps, L-2, gate_right)
    trunc_err_sqr += err_sqr

    # odd layer
    mps, err_sqr = apply_gate_batched(mps, 1, L-1, gate_middle)
    
    ### noise-nonlinear layer ###
    xis = jnp.sqrt(0.5*deltat) * (jr.normal(key,shape=(L,)) + 1j * jr.normal(nk(key),shape=(L,))) + es * deltat * 0.5
    exp_xi = jax.lax.map( lambda xi: expm( xi * jM - 0.5 * deltat * jN),xis  )  
    for i in range(L):
       mps = mps.at[i].set( jnp.einsum( 'ij,pjq->piq',exp_xi[i],mps[i]  )   )
    trunc_err_sqr += err_sqr
    return mps, one_siteE(mps,Op,site)   #opi#trunc_err_sqr norm_squared(mps)
@jit
def rnorm_svd(cr,full=False):
    cr1,cr2,cr3=cr.shape
    s,v,d = svd(cr.reshape(cr1,cr2*cr3))
    d1,d2 = d.shape
    d = d.reshape(d1,cr2,cr3)
    return s,v,d
@jit
def lnorm_svd(cr,full=False):
    cr1,cr2,cr3=cr.shape
    s,v,d = svd(cr.reshape(cr1*cr2,cr3))
    s1,s2 = s.shape
    s = s.reshape(cr1,cr2,s2)
    return s,v,d
@jit
def rnormalize(mps):
    for i in range(0,len(mps)):
        if i+1 < len(mps):
            s,v,d = rnorm_svd(mps[-i-1])
            #v = v/jnp.sqrt(v)
            mps = mps.at[-i-1].set(d)
            mps.at[-i-2].set(jnp.einsum("ijm,mk,k->ijk",mps[-i-2],s,v+0j))
        else:
            cr = jnp.copy(mps[0])
            s,v,d = rnorm_svd(cr)
            mps = mps.at[0].set(d/jnp.sqrt(mps.shape[1]) )
    return mps

@jit
def lnormalize(mps):
    for i in range(0,len(mps)):
            s,v,d = lnorm_svd(mps[i])
            #v = v/jnp.sqrt(v)
            mps = mps.at[i].set(s)
            mps.at[i+1].set(jnp.einsum("k,km,mji->kji",v+0j,d, mps[i+1]  ))
    cr = jnp.copy(mps[len(mps)])
    s,v,d = lnorm_svd(cr)
    mps = mps.at[-1].set(s/jnp.sqrt(mps.shape[1]))
    return mps

if __name__ == '__main__':
    t0, t1 = 0.,10. 
    num_sites = 4 
    chi = 10 
    h = 3.
    mps0 = mps_zero_state(num_sites, chi)
    site =   int(num_sites*0.5)
    deltat = 0.01

    steps = int((t1-t0)/deltat) 
    key =jr.PRNGKey(numpy.random.randint(2**14))
    opis = []
    n_samples = 200
    for _ in range(n_samples):
        key =jr.PRNGKey(numpy.random.randint(2**14))
        _,opi = cp_mps_evolution((h), deltat, steps, mps0,site,key)
        if max(opi)<3:
           opis.append(opi)
    print(jnp.array(opis).shape)
    times = jnp.linspace(t0,t1,steps)

    #plt.plot(times  ,jnp.array(opis).mean( axis = 0), color='tab:red',label='tebd')
    #plt.legend()
    #plt.show()
    key =jr.PRNGKey(numpy.random.randint(2**14))
    psi0  =  jnp.zeros(2**num_sites,dtype=COMPLEX_TYPE)  #basis_state([0 for _ in range(num_sites)])
    psi0  =  psi0.at[-1].set(1.)#(2**num_sites,dtype+COMPLEX_TYPE)  #basis_state([0 for _ in range(num_sites)])
    rho0 =  jnp.einsum('i,j-> ij',psi0.conj(),psi0 )
    Heff = heff(h,num_sites)
    jumps = dissipator(num_sites)
    index = site 
    O = jN
    Oi = jnp.kron(
           jnp.eye(2**index),
           jnp.kron(O, jnp.eye(2**(num_sites-index-1))))
    deltat = 0.01
    solind = solveLindblad(Heff,jumps,num_sites,h,t0,t1,rho0,deltat)
    tr = jnp.einsum( 'tii->t',solind.ys  )#   /jnp.einsum('ti,ti->t', sol.ys,sol.ys.conj()  ).real
    nt = jnp.einsum( 'ij,tji->t',Oi,solind.ys  )#/tr#   /jnp.einsum('ti,ti->t', sol.ys,sol.ys.conj()  ).real
    plt.plot(solind.ts, nt, color='tab:orange',label = 'exact',linestyle = 'dashed')
    plt.plot(times  ,jnp.array(opis).mean( axis = 0), color='tab:red',label='tebd')
    plt.legend()
    plt.show()




