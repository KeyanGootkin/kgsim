import numpy as np
from kgsim import ScalarField, save_dhybridr_scalar_field
from kbasic import Folder

shock = lambda: np.array([10,11,12,13,14,5,6,7,8,9])+np.random.random(10)
n = np.array([np.roll([shock() for i in range(5)], t//30).T for t in range(100)])
bx = np.zeros((100, 10, 5))
by = np.ones((100, 10, 5))
bz = np.ones((100, 10, 5))

if __name__=='__main__':
    save_dhybridr_scalar_field(
        ScalarField(n, verbose=True, debug=True),
        "./simulation/Output/density"
    )
    save_dhybridr_scalar_field(
        ScalarField(bx, verbose=True, debug=True),
        "./simulation/Output/B/x"
    )
    save_dhybridr_scalar_field(
        ScalarField(by, verbose=True, debug=True),
        "./simulation/Output/B/y"
    )
    save_dhybridr_scalar_field(
        ScalarField(bz, verbose=True, debug=True),
        "./simulation/Output/B/z"
    )
