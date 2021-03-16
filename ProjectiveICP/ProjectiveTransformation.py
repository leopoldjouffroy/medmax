import numpy as np

def ProjectiveTransformation(source,target):
    
    # Find & apply the best projective transformation matching the source point cloud to the target point cloud (in homogenous
    # coordinates)
    #
    # Inputs : source : 3*n source point cloud 
    #          target : 3*n target point cloud
    # 
    # Output : y = 3*n point cloud computed by applying the best transformation to the source
    
    n = source.shape[1]
    
    x = target[0,:]
    y = target[1,:]
    X = source[0,:]
    Y = source[1,:]

    rowsXY = -np.ones((3,n))
    rowsXY[0,:] = -X
    rowsXY[1,:] = -Y

    hx = np.zeros((9,n))
    hx[0:3,:] = rowsXY
    hx[6,:] = np.multiply(x,X)
    hx[7,:] = np.multiply(x,Y)
    hx[8,:] = x

    hy = np.zeros((9,n))
    hy[3:6,:] = rowsXY
    hy[6,:] = np.multiply(y,X)
    hy[7,:] = np.multiply(y,Y)
    hy[8,:] = y

    h = np.concatenate((hx,hy),axis=1)
    
    u, s, vh = np.linalg.svd(h)
    v = np.reshape(u[:,8],(3,3))
    
    q = np.dot(v,source)
    p = q[2,:]
    
    # normalization
    xx = np.divide(q[0,:],p)
    yy = np.divide(q[1,:],p)
    
    y = np.ones((3,n))
    y[0,:] = xx
    y[1,:] = yy
    
    return y 