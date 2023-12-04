from PIL import Image, ImageFilter
from typing import Any
import torch as th
import numpy as np
import os

img_ext = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']

class ImageProcessor:
    def __init__(self):
        super().__init__()
        self.img = None
        self.img_pil = None
    
    def extendChannel(self):
        self.checkLoad()
        if len(self.img.shape) == 2:
            self.img = np.expand_dims(self.img, axis=2).astype(np.uint8)
        if self.img.shape[2] == 1:
            self.img = np.repeat(self.img, 3, axis=2).astype(np.uint8)
        self.getPIL(self.img)

    def addGaussianNoise(self, mean, std):
        self.checkLoad()
        var = std**2
        noise = np.random.normal(mean, var, self.img.shape)
        self.img = np.clip(self.img + noise, 0, 255).astype(np.uint8)
        self.getPIL(self.img)
        return self

    def addSaltAndPepperNoise(self, prob):
        self.checkLoad()
        if prob > 1 or prob < 0:
            raise ValueError("Probability must be between 0 and 1")
        random_image_arr = np.random.choice(
                                            [0, 1, np.nan], 
                                            p = [prob / 2, 1 - prob, prob / 2], 
                                            size = self.img.shape
                                            )
        modified_img = self.img.astype(np.float32) * random_image_arr
        self.img = np.nan_to_num(modified_img, nan = 255).astype(np.uint8)
        self.getPIL(self.img)
        return self

    def addThreshold(self, threshold_scale: float):
        self.checkLoad()
        self.normalize()
        self.img = np.where(self.img > threshold_scale*self.img.max(), self.img.max(), self.img.min())
        self.getPIL(self.img)
        return self
    
    def normalize(self):
        self.checkLoad()
        self.img = np.clip(self.img - self.img.mean() + self.img.min(), 0, 255).astype(np.uint8)
        self.getPIL(self.img)
        return self

    def downAndUpSample(self, scale: float):
        self.checkLoad()
        self.rescale(scale)
        self.resize((self.bh, self.bw))
        return self

    def addMotionBlur(self):
        self.checkLoad()
        self.img_pil = self.img_pil.filter(ImageFilter.BLUR)
        self.getNumpy(self.img_pil)
        return self

    def resize(self, size: tuple):
        self.checkLoad()
        self.img_pil = self.img_pil.resize(size, Image.BICUBIC)
        self.getNumpy(self.img_pil)
        return self

    def rescale(self, scale: float):
        self.checkLoad()
        new_height, new_width = int(self.bh*scale), int(self.bw*scale)
        self.resize((new_height, new_width))
        self.getNumpy(self.img_pil)
        return self

    def load_from_path(self, path: str):
        self.img_pil = Image.open(path)
        self.getNumpy(self.img_pil)

    def load_from_tensor(self, tensor: th.Tensor):
        self.img = tensor
        if tensor.shape[0] == 1:
            self.img = self.img.squeeze(0)
        self.img.permute(1, 2, 0)
        self.getNumpy(self.img)
        self.getPIL(self.img)

    def load_from_numpy(self, arr: np.ndarray):
        self.img = arr
        self.getPIL(self.img)
    
    def getNumpy(self, arr):
        self.img = np.array(arr)
        self.getPIL(self.img)

    def getTensor(self):
        self.checkLoad()
        return th.from_numpy(self.img).permute(2, 0, 1)
    
    def getPIL(self, arr):
        self.img_pil = Image.fromarray(arr)

    def save(self, path, name, ext = 'png'):
        self.checkLoad()
        self.checkPath(path)
        if ext[1:] not in img_ext:
            raise ValueError("Extension must be one of the following: " + str(img_ext))
        self.img_pil.save(path+name+'.'+ext)

    def show(self):
        self.checkLoad()
        self.img_pil.show()
    
    def load(self, loc: Any):
        if isinstance(loc, str):
            self.load_from_path(loc)
        elif isinstance(loc, th.Tensor):
            self.load_from_tensor(loc)
        elif isinstance(loc, np.ndarray):
            self.load_from_numpy(loc)
        elif isinstance(loc, Image.Image):
            self.getNumpy(loc)
        else:
            raise TypeError("Input must be a path, tensor, PIL Image or ndarray")
        
        self.extendChannel() 
        
        self.bh, self.bw = self.img.shape[0], self.img.shape[1]
        return self
    
    def checkLoad(self):
        if self.img is None:
            raise ValueError("Image is not loaded yet")
    
    def checkPath(self, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
