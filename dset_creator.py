from skimage import io
from tqdm import tqdm
import numpy as np

from custom_image import ImageProcessor as IP

def createDataset(
                    load_path: str, 
                    save_path: str = None, 
                    ext: str = None, 
                    resize: tuple[int, int] = None,
                    rescale: float = None,
                    gaussian_noise: tuple[float, float] = None,
                    normalize: bool = False,
                    downAndUpSample: float = None,
                    addMotionBlur: bool = False,
                    show: bool = False
                    ):
    for i,img in enumerate(tqdm(io.imread(load_path).astype(np.uint8))):
        obj = IP().load(img)
        if normalize:
            obj.normalize()
        if gaussian_noise is not None:
            mean, std = gaussian_noise
            obj.addGaussianNoise(mean, std)
        if downAndUpSample is not None:
            obj.downAndUpSample(downAndUpSample)
        if addMotionBlur:
            obj.addMotionBlur()
        if rescale is not None:
            obj.rescale(rescale)
        if resize is not None:
            obj.resize(resize)
        if save_path is not None and ext is not None:
            obj.save(save_path, str(i), '.' + ext)
        if show:
            obj.show()

if __name__ == '__main__':
    load_path = '../Pristine/PTY_pristine_raw.tif'
    save_path = './data/'
    ext = 'png'
    createDataset(
                    load_path, 
                    save_path, 
                    ext, 
                    normalize = True,
                    downAndUpSample=0.5,
                    addMotionBlur = True
                    )
    print('Done.')