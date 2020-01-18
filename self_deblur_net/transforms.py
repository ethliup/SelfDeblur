import torch
import random
import numbers
import collections
import numpy as np

class Merge(object):
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray) for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")

class Normalize(object):
    def __call__(self, image):
        image=image/255.
        return image

class Crop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        
        th, tw = self.size
        if w == tw and h == th:
            return img

        return img[:th, :tw, :]

class Random_crop(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        random.seed()

    def __call__(self, img):
        h, w = img.shape[:2]
        
        th, tw = self.size
        if w == tw and h == th:
            return img

        w1 = 0 if w==tw else random.randint(0, w - tw)
        h1 = 0 if h==th else random.randint(0, h - th)

        return img[h1:h1+th, w1:w1+tw, :]

class Split(object):
    """Split images into individual arraies
    """
    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            slices_.append(s)
        self.slices = slices_

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                ret.append(image[:, :, s[0]:s[1]])
            return ret
        else:
            raise Exception("obj is not an numpy array")

class To_tensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img
