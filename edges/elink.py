import numpy as np
import math


def edge_link(edge, mag, ori):
    '''Use hysteresis to link edges based on high and low magnitude thresholds
        - Input edge: H x W binary////logical map after non-max suppression
        - Input mag: H x W matrix represents the magnitude of gradient
        - Input ori: H x W matrix represents the orientation of gradient
        - Output linked_edge: H x W binary matrix represents the final canny edge detection map
    '''
    edge = np.squeeze(np.asarray(edge))
    mag = np.squeeze(np.asarray(mag))
    muse = edge * mag

    # Hysteresis thresholding
    # for weak edge
    threshold_low = 0.015  # TODO
    threshold_low = threshold_low * muse.max()
    # for strong edge
    threshold_high = 0.115 # TODO
    threshold_high = threshold_high * muse.max()

    if np.sum(muse)<50000:
        threshold_low = 0.01
        threshold_high = 0.02
        threshold_low = threshold_low * muse.max()
        threshold_high = threshold_high * muse.max()
    elif np.sum(muse)<250000 and np.sum(threshold_low)>2:
        threshold_low = 0.02
        threshold_high = 0.04
        threshold_low = threshold_low * muse.max()
        threshold_high = threshold_high * muse.max()
    elif np.sum(muse)<550000 and np.sum(threshold_low)>2:
        threshold_low = 0.015
        threshold_high = 0.06
        threshold_low = threshold_low * muse.max()
        threshold_high = threshold_high * muse.max()

    size = mag.shape
    linked_edge = np.zeros(size)

    is_weak = muse>threshold_low
    is_strong = muse>threshold_high 
    
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            if linked_edge[i,j]==0:
                if is_strong[i,j]:
                    linked_edge[i,j]=1
                    if ori[i,j]==0:
                        if (is_weak[i+1,j]):
                            linked_edge[i+1,j]=1 
                        if (is_weak[i-1,j]):
                            linked_edge[i-1,j]=1
                    elif ori[i,j]>0 and ori[i,j]<math.pi/4:
                        if (is_weak[i+1,j]):
                            linked_edge[i+1,j]=1 
                        if (is_weak[i-1,j]):
                            linked_edge[i-1,j]=1
                        if (is_weak[i+1,j-1]):
                            linked_edge[i+1,j-1]=1
                        if  (is_weak[i-1,j+1]):
                            linked_edge[i-1,j+1]=1
                    elif ori[i,j]==math.pi/4:
                        if (is_weak[i+1,j-1]):
                            linked_edge[i+1,j-1]=1
                        if  (is_weak[i-1,j+1]):
                            linked_edge[i-1,j+1]=1
                    elif ori[i,j]>math.pi/4 and ori[i,j]<math.pi/2:
                        if (is_weak[i+1,j-1]):
                            linked_edge[i+1,j-1]=1
                        if  (is_weak[i-1,j+1]):
                            linked_edge[i-1,j+1]=1
                        if  (is_weak[i,j+1]):
                            linked_edge[i,j+1]==1
                        if (is_weak[i,j-1]):
                            linked_edge[i,j-1]==1
                    elif ori[i,j]==math.pi/2:
                        if  (is_weak[i,j+1]):
                            linked_edge[i,j+1]==1
                        if (is_weak[i,j-1]):
                            linked_edge[i,j-1]==1
                    elif ori[i,j]>math.pi/2 and ori[i,j]<math.pi*3/4:
                        if  (is_weak[i,j+1]):
                            linked_edge[i,j+1]==1
                        if (is_weak[i,j-1]):
                            linked_edge[i,j-1]==1
                        if (is_weak[i+1,j+1]):
                            linked_edge[i+1,j+1]=1
                        if (is_weak[i-1,j-1]):
                            linked_edge[i-1,j-1]=1
                    elif ori[i,j]==math.pi*3/4:
                        if (is_weak[i+1,j+1]):
                            linked_edge[i+1,j+1]=1
                        if (is_weak[i-1,j-1]):
                            linked_edge[i-1,j-1]=1
                    elif ori[i,j]>math.pi*3/4:
                        if (is_weak[i+1,j+1]):
                            linked_edge[i+1,j+1]=1
                        if (is_weak[i-1,j-1]):
                            linked_edge[i-1,j-1]=1
                        if (is_weak[i+1,j]):
                            linked_edge[i+1,j]=1 
                        if (is_weak[i-1,j]):
                            linked_edge[i-1,j]=1
                                            
    return linked_edge


if __name__ == '__main__':
    from PIL import Image
    from derivatives import get_derivatives
    from nms import non_max_suppression
    gray = Image.open("../Madison.png").convert('L')
    mag, *_, ori = get_derivatives(gray, sigma=1.0)
    edge = non_max_suppression(mag, ori, threshold=3)
    linked_edge = edge_link(edge, mag, ori)
