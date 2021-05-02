import math
import numpy as np
from scipy import signal


def gaussian_pdf_1d(mu, sigma, length):
    '''Generate one dimension Gaussian distribution
        - input mu: the mean of pdf
        - input sigma: the standard derivation of pdf
        - input length: the size of pdf
        - output: a row vector represents one dimension Gaussian distribution
    '''
    # create an array
    half_len = length / 2

    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

    ax = ax.reshape([-1, ax.size])
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

    return nominator / denominator

def gaussian_pdf_2d(mu, sigma, row, col):
    '''Generate two dimensional Gaussian distribution
        - input mu: the mean of pdf
        - input sigma: the standard derivation of pdf
        - input row: length in row axis
        - input column: length in column axis
        - output: a 2D matrix represents two dimensional Gaussian distribution
    '''
    # create row vector as 1D Gaussian pdf
    g_row = gaussian_pdf_1d(mu, sigma, row)
    # create column vector as 1D Gaussian pdf
    g_col = gaussian_pdf_1d(mu, sigma, col).transpose()

    return signal.convolve2d(g_row, g_col, mode='full')


def get_derivatives(gray, sigma=0.4):
    '''Compute gradient information of the input grayscale image
        - Input gray: H x W matrix as image
        - Output mag: H x W matrix represents the magnitude of derivatives
        - Output magx: H x W matrix represents the magnitude of derivatives along x-axis
        - Output magy: H x W matrix represents the magnitude of derivatives along y-axis
        - Output ori: H x W matrix represents the orientation of derivatives
    '''
    mu = 0
    sigma = sigma  # 0.4, less sigma, more blurred edge
    Ga = gaussian_pdf_2d(mu, sigma, 5, 5)

    # Filter 
    dx = np.array([[1, 0, -1]])  # Horizontal
    dy = np.array([[1], [0], [-1]])  # Vertical
    #dx = np.array([[1, -1]]) # Horizontal
    #dy = np.array([[1],[-1]]) # Vertical
    
    # Convolution of image
    #Gx = np.convolve(Ga, dx, 'same')
    #Gy = np.convolve(Ga, dy, 'same')
    #lx = np.convolve(I_gray, Gx, 'same')
    #ly = np.convolve(I_gray, Gy, 'same')

    Gx = signal.convolve2d(Ga, dx, mode='same', boundary='fill')
    Gy = signal.convolve2d(Ga, dy, mode='same', boundary='fill')
    lx = signal.convolve2d(gray, Gx, mode='same', boundary='fill')
    ly = signal.convolve2d(gray, Gy, mode='same', boundary='fill')

    # Magnitude
    mag = np.sqrt(lx*lx+ly*ly)

    # Angle
    angle = np.arctan(ly/lx)
    angle[angle<0] = math.pi + angle[angle<0]
    angle[angle>7*math.pi/8] = math.pi - angle[angle>7*math.pi/8]

    ## Edge angle discretization into 0, pi/4, pi/2, 3*pi/4
    #angle[angle>=0 and angle<math.pi/8] = 0
    #angle[angle>=math.pi/8 and angle<3*math.pi/8] = math.pi/4
    #angle[angle>=3*math.pi/8 and angle<5*math.pi/8] = math.pi/2
    #angle[angle>=5*math.pi/8 and angle<=7*math.pi/8] = 3*math.pi/4

    return mag, lx, ly, angle


if __name__ == '__main__':
    from PIL import Image
    gray = Image.open("../Madison.png").convert('L')
    mag, magx, magy, ori = get_derivatives(gray)
