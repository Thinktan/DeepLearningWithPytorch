import collections
import copy
import datetime
import gc
import time

# import torch
import numpy as np

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

'''
xyz -> cri
x -> column
y -> row
z -> index
'''

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    '''

    :param coord_irc: 要转换的IRC坐标
    :param origin_xyz: XYZ坐标系中的原点
    :param vxSize_xyz: 体素大小
    :param direction_a: 方向矩阵，用于处理影像方向
    :return:
    '''
    cri_a = np.array(coord_irc)[::-1] # 转numpy，反转顺序

    origin_a = np.array(origin_xyz) # 转numpy
    vxSize_a = np.array(vxSize_xyz)

    # cri_a * vxSize_a 首先将IRC坐标乘以体素大小，将其转换为物理空间尺寸。
    # 然后通过矩阵 direction_a 乘以这个结果，来考虑影像的方向。最后加上原点 origin_a。
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    '''

    :param coord_xyz: 要转换的XYZ坐标
    :param origin_xyz: XYZ坐标系中的原点
    :param vxSize_xyz: 体素大小
    :param direction_a: 方向矩阵，用于考虑影像的方向
    :return:
    '''
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)

    # (coord_a - origin_a) 将 XYZ 坐标相对于原点进行调整。
    # 然后，使用 @（矩阵乘法）将这个调整后的坐标与方向矩阵的逆矩阵 np.linalg.inv(direction_a) 相乘。
    # 这一步是必要的，因为影像的方向可能与标准的 XYZ 坐标系不同。
    # 最后，将结果除以体素大小 vxSize_a，将物理尺寸转换回 IRC 坐标。
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module