# reference: https://github.com/Andrew-Qibin/dss_crf/blob/master/examples/dense_hsal.py
import torch
import numpy as np
import pydensecrf.densecrf as dcrf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# parameter
EPSILON = 1e-8
tau = 1.05


# img: PIL
# anno: numpy
def crf(img, anno, to_tensor=False):
    img = np.array(img)
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    n_energy = -np.log((1.0 - anno + EPSILON)) / (tau * sigmoid(1 - anno))
    p_energy = -np.log(anno + EPSILON) / (tau * sigmoid(anno))
    U = np.zeros((2, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = np.expand_dims(infer[1, :].reshape(img.shape[:2]), 0)
    if to_tensor:
        res = torch.from_numpy(res).unsqueeze(0)
    return res
