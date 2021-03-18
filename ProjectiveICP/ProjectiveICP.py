import meshio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import json

# Homemade
from ProjectiveTransformation import ProjectiveTransformation
from RigidTransformation import RigidTransformation

# Verbose
verbose = True

# Functions
##################################################################################################

def normalization(mesh):
    maxX = np.amax(mesh[0,:]) 
    minX = np.amin(mesh[0,:]) 
    maxY = np.amax(mesh[1,:]) 
    minY = np.amin(mesh[1,:])
    maxZ = np.amax(mesh[2,:]) 
    minZ = np.amin(mesh[2,:]) 

    mesh[0,:] /= (maxX-minX)
    mesh[1,:] /= (maxY-minY)
    mesh[2,:] /= (maxZ-minZ)

    return mesh

def nearest_neighbors(source,target):
    neigh = NearestNeighbors(n_neighbors=1,algorithm='auto',metric='euclidean')
    neigh.fit(target.T)
    distances, indices = neigh.kneighbors(source.T, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_transformation(source,target,transformation):
    if transformation == 'rigid':
        source = RigidTransformation(source,target)
    elif transformation == 'projective':
        source = ProjectiveTransformation(source,target)
    return source

def icp(source,target,transformation,fig,max_iterations=15,tolerance = 0.0001):

    prev_error = 0

    # ICP loop
    for i in range(max_iterations):

        print(transformation," : ",i)
        # find the nearest neighbors
        distances, indices = nearest_neighbors(source, target)
        
        # Compute & apply the best transformation
        source = best_transformation(source,target[:,indices],transformation)

        # Convergence check
        mean_error = np.mean(distances)
        
        if np.abs(prev_error - mean_error) < tolerance:
                break
        prev_error = mean_error

        # display
        plt.figure(fig)
        plt.clf()
        plt.axis([-1, 1, -1, 1])
        plt.plot(target[0,:],target[1,:],'b+')
        plt.plot(source[0,:],source[1,:],'r+')
        plt.pause(0.05)

    plt.show()

    return source

##################################################################################################

# Data
source_mesh = meshio.read(r"C:\Users\Medmax\Documents\Bianca\Meshes\Mandibles\off\ferrier denise mandibule.off",file_format='off')
source_points = source_mesh.points.T

target_mesh = meshio.read(r"C:\Users\Medmax\Documents\Bianca\Meshes\Mand.off",file_format='off')
target_points = target_mesh.points.T

# normalization
source_points = normalization(source_points)
target_points = normalization(target_points)

# compute the centroids
n = source_points.shape[1]
m = target_points.shape[1]
centroid_source_points = np.mean(source_points, axis=1)
centroid_target_points = np.mean(target_points, axis=1)

# translate points to their centroids
source_points = source_points - np.vstack(centroid_source_points)
target_points = target_points - np.vstack(centroid_target_points)


#target_points[2,:] *= -1 
copy_mesh = np.copy(source_points)


# Première vue
##################################################################################################
source = copy_mesh[[0,1],:]
target = target_points[[0,1],:]

# homogeneous coordinates
source = np.vstack((source,np.ones((1,n))))
target = np.vstack((target,np.ones((1,m))))

# display
if(verbose):
    plt.figure(1)
    plt.axis([-1, 1, -1, 1])
    plt.plot(target[0,:],target[1,:],'b+')
    plt.plot(source[0,:],source[1,:],'r+')
    plt.show()

# ICP Rigide
source = icp(source,target,'rigid',4)

# ICP Projective
source = icp(source,target,'projective',5)

# Update copy mesh
copy_mesh[[0,1],:] = source[[0,1],:]

##################################################################################################

# Seconde vue
##################################################################################################
source = copy_mesh[[0,2],:]
target = target_points[[0,2],:]

# display the centered meshes
if(verbose):
    plt.figure(8)
    plt.axis([-1, 1, -1, 1])
    plt.plot(target[0,:],target[1,:],'b+')
    plt.plot(source[0,:],source[1,:],'r+')
    plt.show()

# homogeneous coordinates
source = np.vstack((source,np.ones((1,n))))
target = np.vstack((target,np.ones((1,m))))

# ICP Rigide
source = icp(source,target,'rigid',9)

# ICP Projective
source = icp(source,target,'projective',10)

# Update copy mesh
copy_mesh[[0,2],:] = source[[0,1],:]

##################################################################################################

# Troisième vue
##################################################################################################
source = copy_mesh[[1,2],:]
target = target_points[[1,2],:]

# display the centered meshes
if(verbose):
    plt.figure(13)
    plt.axis([-1, 1, -1, 1])
    plt.plot(target[0,:],target[1,:],'b+')
    plt.plot(source[0,:],source[1,:],'r+')    
    plt.show()

# homogeneous coordinates
source = np.vstack((source,np.ones((1,n))))
target = np.vstack((target,np.ones((1,m))))

# ICP Rigide
source = icp(source,target,'rigid',14)

# ICP Projective
#source = icp(source,target,'projective',15)

# Update copy mesh
copy_mesh[[1,2],:] = source[[0,1],:]
##################################################################################################

meshio.Mesh(copy_mesh.T,source_mesh.cells).write("foo.off",file_format="off")
meshio.Mesh(source_points.T,source_mesh.cells).write("source.off",file_format="off")
meshio.Mesh(target_points.T,target_mesh.cells).write("target.off",file_format="off")








