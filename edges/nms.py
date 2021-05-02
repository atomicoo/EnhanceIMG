import numpy as np
from scipy import interpolate
try:
    from .interp import interp2d
except ImportError:
    from interp import interp2d


def non_max_suppression(mag, ori, threshold=5):
    '''Find local maximum edge pixel using NMS along the line of the gradient
        - Input mag: H x W matrix represents the magnitude of derivatives
        - Input ori: H x W matrix represents the orientation of derivatives
        - Output edge: H x W binary matrix represents the edge map after non-maximum suppression
    '''
    # suppress all the gradient values to 0 except the local maximal
    size = mag.shape

    # interpolation
    x = np.arange(1, size[0]+1, 1)
    y = np.arange(1, size[1]+1, 1)
    xx, yy = np.meshgrid(y, x)
    #f = interpolate.interp2d(xx, yy, mag)
    #f = interpolate.Rbf(xx, yy, mag)

    xnew1 = xx + np.cos(ori)
    ynew1 = yy + np.sin(ori)
    xnew2 = xx - np.cos(ori)
    ynew2 = yy - np.sin(ori)
    #znew1 = f(xnew1, ynew1)
    znew1 = interp2d(xx, yy, mag, xnew1, ynew1)
    #znew2 = f(xnew2, ynew2)
    znew2 = interp2d(xx, yy, mag, xnew2, ynew2)
    
    edge1 = np.zeros(size)
    edge2 = np.zeros(size)

    edge1[mag >= znew1+threshold] = 1
    edge2[mag >= znew2+threshold] = 1
    edge = edge1 * edge2
    edge = np.matrix(edge)

    return edge


if __name__ == '__main__':
    from PIL import Image
    from derivatives import get_derivatives
    gray = Image.open("../Madison.png").convert('L')
    mag, *_, ori = get_derivatives(gray)
    edge = non_max_suppression(mag, ori)
