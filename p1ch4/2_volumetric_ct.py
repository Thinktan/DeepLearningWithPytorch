import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)

import imageio

# 体数据 c*d*h*w

dir_path = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')
print(vol_arr.shape)

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

print(vol.shape)

import matplotlib.pyplot as plt

plt.imshow(vol_arr[90])
plt.show()