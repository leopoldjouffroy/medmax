import numpy as np

def RigidTransformation(source,target):
    
    # Compute the rotation matrix
    H = np.dot(source[0:2,:],target[0:2,:].T)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Complete R so it is expressed in homogeneous coordinates
    R = np.vstack((R,np.zeros((1,2))))
    R = np.hstack((R,np.zeros((3,1))))
    R[2,2] = 1
    
    # Apply transformation
    result = np.dot(R,source)
    
    return result