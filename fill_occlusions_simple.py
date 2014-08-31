
# In[1]:

import numpy
import scipy
import matplotlib.pyplot as plt


# In[2]:

from scipy import ndimage
import numpy as np

# kernels for shift
k = np.array([
[[0,0,0],
 [0,0,1],
 [0,0,0],],
[[0,1,0],
 [0,0,0],
 [0,0,0],],
[[0,0,0],
 [1,0,0],
 [0,0,0],],
[[0,0,0],
 [0,0,0],
 [0,1,0],],
[[1,0,0],
 [0,0,0],
 [0,0,0],],
[[0,0,1],
 [0,0,0],
 [0,0,0],],
[[0,0,0],
 [0,0,0],
 [0,0,1],],
[[0,0,0],
 [0,0,0],
 [1,0,0],],
])



# In[8]:

from scipy import misc

from PIL import Image

#reading source data
frm = misc.imread('cones_2_6_align2_00000.png')
dpt0 = misc.imread('disp2.png')
occl0 = misc.imread('occl.png')

#initial values for results
dpt = dpt0
occl = occl0

frm_shifterd = numpy.zeros((len(k),)+frm.shape)
dif = numpy.zeros((len(k),)+frm.shape)
sumdiff = numpy.zeros((len(k),)+frm.shape[:-1])

for ik in range(len(k)):
    for ch in range(frm.shape[-1]):
        frm_shifterd[ik,:,:,ch] = ndimage.convolve(frm[:,:,ch], k[ik], mode='nearest', cval=0.0)
    dif[ik] = numpy.abs(frm_shifterd[ik,:,:,:]-frm[:,:,:]) 
    sumdiff[ik] = dif[ik].sum(axis=2) 
    

iteration = 0


def one_iteration(dpt, occl, k, sumdiff):
    dpt_shifted = numpy.zeros((len(k),)+dpt.shape)
    occl_shifted = numpy.zeros((len(k),)+dpt.shape)
    sumdiff_final = numpy.zeros((len(k),)+frm.shape[:-1])

    
    occl_dil = ndimage.grey_dilation(occl, size=(3,3))
    for ik in range(len(k)):
        dpt_shifted[ik,:,:] = ndimage.convolve(dpt[:,:], k[ik], mode='nearest', cval=0.0)
    
        occl_shifted[ik,:,:] = ndimage.convolve(occl[:,:], k[ik], mode='nearest', cval=0.0)
        sumdiff_final[ik] = sumdiff[ik] + (1-occl_shifted[ik])*1000000000
        #print 'shifted', ik, '\n', dpt_shifted[ik]
        
    # chose direction where the difference is the lowest
    good_directions = numpy.argmin(sumdiff_final, axis = 0)
    dpt_best = numpy.choose( good_directions, dpt_shifted )
    
    dpt_new = numpy.choose( occl==occl_dil, numpy.array([dpt_best,dpt]))
    
    return dpt_new, occl_dil

while True:
    iteration += 1
    
    dpt_new, occl_dil = one_iteration(dpt, occl, k, sumdiff)
    
    if numpy.array_equal(occl, occl_dil):
        break;
    
    #Image.fromarray(np.cast['uint8'](dpt_new)).save('dpt_{}.png'.format(iteration))

        
    dpt = numpy.copy(dpt_new)
    occl = numpy.copy(occl_dil)
    
#write output 
Image.fromarray(np.cast['uint8'](dpt)).save('processed_dpt.png')



