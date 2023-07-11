from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    dataset= np.load(filename, mmap_mode=None)
    mean= np.mean(dataset,axis=0)
    return dataset-mean

def get_covariance(dataset):
    cov= np.dot(np.transpose(dataset), dataset)
    return cov/(len(dataset)-1)
    # Your implementation goes here!
    pass

def get_eig(S, m):
    # Your implementation goes here!
    mLargestVals, mlargestvec=eigh(S,subset_by_index=[len(S)-m,len(S)-1]) #i think this does it in ascending order
    #descending=np.flip(mLargestVals)
    #print(descending)
    #print(mlargestvec)
    #print(np.argsort(mLargestVals)) #print 0,1
    i=np.flip(np.argsort(mLargestVals)) #prints 1,0
    #why does it help if we have the ranking rather than the value?... why not the code below
    #i = np.flip(mLargestVals)
    return np.diag(mLargestVals[i]),mlargestvec[:,i]

    pass

def get_eig_prop(S, prop):
    # Your implementation goes here!
    #find the val you need to get higher than >>sum of eigvals * prop
    #
    mVals, mvec = eigh(S)
    minval= np.sum(mVals)*prop
    mPropVals, mPropvec = eigh(S, subset_by_value=[minval ,np.inf])  # i think this does it in ascending order
    i = np.flip(np.argsort(mPropVals))  # prints 1,0
    return np.diag(mPropVals[i]), mPropvec[:, i]

    pass

def project_image(image, U):
    # Your implementation goes here!
    #find how many eigvec U has
    #print(len(U)) #1024 .. image xi projected into a 1024 dimensional subspace
    #The PCA projection represents images as a weighted sum of the eigenvectors.
    # ^weighted sum of U
    # This projection only needs to store the weight for each
    # eigenvector (m-dimensions) instead of the entire image (d-dimensions).
    #needs to store weight for each val in U
    #formula: sum from j=1 to m OF aij*uj
    #each eigenvector uj is multiplied by its corresponding weight αij .
    #how to get aij? >>  transpose u and mult. by image
    #do dot product cause 'weighted sum'
    aij=np.dot(np.transpose(U),image)
    pca= np.dot(U, aij) #why this order? prob something about the dimensions
    return pca
    pass

def display_image(orig, proj):
    # Your implementation goes here!
    #display_image(x[0], projection)
    #1. Reshape the images to be 32 × 32 (before 1 dimensional vectors in R 1024).
    projmodif=np.transpose(proj.reshape(32,32))
    origmodif=np.transpose(orig.reshape(32,32))
    #2. Create a figure with one row of two subplots.
    fig, (pl1, pl2)=plt.subplots(nrows=1, ncols=2)
    #3. Title the first subplot (the one on the left) as “Original” (without the quotes) and the second (the one
    # on the right) as “Projection” (also without the quotes).
    pl1.set_title("Original")
    pl2.set_title("Projection")
    #4.Use imshow with the optional argument aspect='equal'
    imshoworig= pl1.imshow(origmodif, aspect='equal')
    imshowproj=pl2.imshow(projmodif,aspect='equal')
    #5.Use the return value of imshow to create a colorbar for each image.
    fig.colorbar(imshoworig,ax=pl1)
    fig.colorbar(imshowproj, ax=pl2)
    #6. Render your plots!
    #pl1.savefig()
    #pl2.savefig()
    plt.show()
    pass
# x=load_and_center_dataset("YaleB_32x32.npy")
# print(x)
# S=get_covariance(x)
# print(S)
# Lambda,U =get_eig(S, 2)
# print(Lambda,U)
# Lambda,U =get_eig_prop(S, 0.07)
# print(Lambda,U)
# projection = project_image(x[0], U)
# print(projection)
# display_image(x[0], projection)