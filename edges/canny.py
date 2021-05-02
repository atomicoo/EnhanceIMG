import cv2
try: 
    from .derivatives import get_derivatives
    from .nms import non_max_suppression
    from .elink import edge_link
except ImportError:
    from derivatives import get_derivatives
    from nms import non_max_suppression
    from elink import edge_link


def canny_edge(img, sigma=0.4, threshold=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mag, *_, ori = get_derivatives(gray, sigma=sigma)
    edge = non_max_suppression(mag, ori, threshold=threshold)
    linked_edge = edge_link(edge, mag, ori)
    return linked_edge, edge

if __name__ == '__main__':
    import numpy as np
    img = cv2.imread("../Madison.png")
    linked_edge, edge = canny_edge(img, 1.0, 3)
    cat_img = np.concatenate([edge, linked_edge], axis=1)
    cv2.imshow("Edge", cat_img)
    cv2.waitKey(0)
