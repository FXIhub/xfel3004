import numpy as np
from scipy.special import jn

def radial_avg_stack(in_data, in_indices):
    """
    Assumes in_data is a 3 dimensional numpy object.
    in_indices could be either 2-dimensional or 3-dimensional object.
    """
    num_pulses = in_data.shape[-1]
    lin_arr = np.zeros((num_pulses, in_indices.max()+1))
    if len(in_indices.shape) == 3:
        wts_arr = np.zeros((num_pulses, in_indices.max()+1))
        for i in range(num_pulses):
            np.add.at(wts_arr[i], in_indices[:,:,i].ravel(), 1.)
            np.add.at(lin_arr[i], in_indices[:,:,i].ravel(), in_data[:,:,i].ravel())
    else:
        wts_arr = np.zeros(in_indices.max()+1)
        np.add.at(wts_arr, in_indices.ravel(), 1.)
        for i in range(num_pulses):
            np.add.at(lin_arr[i], in_indices.ravel(), in_data[:,:,i].ravel())
    wts_arr += (np.fabs(wts_arr)<1.E-6)
    return lin_arr/(wts_arr)


def gen_indices(num_pulses=50, indexDim=2, cx=10, cy=10, dx=128, dy=512):
    xl = np.linspace(cx, dx+cx, dx, endpoint=False)
    yl = np.linspace(cy, dy+cy, dy, endpoint=False)
    x,y = np.meshgrid(xl, yl)
    indices = np.sqrt(x**2 + y**2)
    indices = np.repeat(indices, num_pulses).reshape(indices.shape+(-1,))
    if indexDim == 2:
        indices = indices.mean(axis=2).astype(int)
    else:
        indices = indices.astype(int)
    return indices

def gen_data(num_pulses=50, cx=10, cy=10, dy=128, dx=512):
    xl = np.linspace(-cx, dx-cx, dx, endpoint=False)
    print(xl.shape)
    yl = np.linspace(-cy, dy-cy, dy, endpoint=False)
    x,y = np.meshgrid(xl, yl)
    print(x.shape)
    # data = np.repeat(np.sin(np.pi*np.sqrt((x)**2 + (y)**2)/dx), num_pulses).reshape(dx, dy, num_pulses)
    # data = np.repeat(np.sin(np.pi*np.sqrt((x)**2 + (y)**2)/dx), num_pulses).transpose(dx, dy, num_pulses)
    data = np.zeros((num_pulses, dy, dx))
    data[:] = np.sin(np.pi*np.sqrt((x)**2 + (y)**2)/dx*8)
    data = data.transpose((1, 2, 0))
    print(data.shape)
    data += 10.*np.random.rand(*data.shape)
    return data

def ball_radial_intensity(fluence, size, pixel):
    p = pixel * size
    return fluence * p ** (-3) * jn(1.5, p) **2
