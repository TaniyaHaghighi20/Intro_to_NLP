import PIL.Image
import numpy as np
import torchvision.transforms.v2.functional as TF
from PIL import ImageOps, ImageFilter, ImageDraw
from numpy import random
import torchvision.transforms.v2 as T
import random


def compute_cdf(hist):
    # Calculate the cumulative distribution function from the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())  # Normalize
    return cdf_normalized


    # Calculate the histogram for the image
    hist_img = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    cv2.normalize(hist_img, hist_img)

    # Compute the CDF for image and reference
    cdf_img = compute_cdf(hist_img)
    cdf_ref = compute_cdf(ref_hist)

    # Create a lookup table
    lookup_table = np.zeros(256)
    for i in range(256):
        diff_cdf = np.abs(cdf_ref - cdf_img[i])
        closest_index = np.argmin(diff_cdf)
        lookup_table[i] = closest_index

    # Apply the mapping to the image
    matched_image = cv2.LUT(image, lookup_table)
    return matched_image


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, crop_size=(400, 400), resize=(128, 128), nopad=True):

        # if isinstance(crop_size, numbers.Number):
        #     self.size = (int(crop_size), int(crop_size))
        # else:
        #     self.size = crop_size
        self.crop_size = crop_size
        self.resize = resize
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, centroid=None):
        w, h = img.size
        # ASSUME H, W
        th, tw = self.crop_size
        if w == tw and h == th:
            return img

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)).resize(size=self.resize, resample=PIL.Image.LANCZOS)


class SobelFilter():
    def __call__(self, img):
        img_x = img.filter(ImageFilter.FIND_EDGES)
        img_y = img.transpose(
            PIL.Image.FLIP_LEFT_RIGHT).filter(ImageFilter.FIND_EDGES).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return PIL.Image.blend(img_x, img_y, alpha=0.5)


class CustomRotation:
    def __init__(self, angles=None):
        if angles is None:
            angles = [0, 90, 180, 270]
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle, expand=True)


class UnsharpMaskTransform(object):
    def __init__(self, radius=2, percent=150, threshold=3):
        """
        Initialize the Unsharp Mask filter parameters.
        :param radius: The radius of the blur filter.
        :param percent: The percentage of the edge enhancement.
        :param threshold: The threshold for the filter.
        """
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, img):
        """
        Apply the Unsharp Mask filter to the input image.
        :param img: PIL image to be sharpened.
        :return: Sharpened PIL image.
        """
        return img.filter(ImageFilter.UnsharpMask(
            radius=self.radius, percent=self.percent, threshold=self.threshold))

class ZScoreNormalization:
    """Apply Z-score normalization to an image."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image to be normalized.

        Returns:
            Tensor: Z-score normalized image.
        """
        if self.mean is None or self.std is None:
            # Calculate mean and std if not provided
            self.mean = img.mean()
            self.std = img.std()
        return (img - self.mean) / (self.std + 1e-8)  # Adding a small value to avoid division by zero

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)





def rotation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        CustomRotation(angles=[90, 180, 270]),
        T.Grayscale(3),
    ])


def colorJitter(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        T.ElasticTransform(alpha=(50.0, 250.0), sigma=(5.0, 10.0)),
        T.Grayscale(3),
    ])


def sobelFilter(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        SobelFilter(),
        T.Grayscale(3),
    ])


def gaussianBlur(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        T.GaussianBlur(kernel_size=int(5), sigma=(0.25, 1.75)),
        T.Grayscale(3),
    ])


def to_PIL():
    return T.Compose([T.ToPILImage(),
                      T.Grayscale(3)
                      ])


def to_Tensor():
    return T.ToTensor()


def ensure_three_channels(img):
    """Convert an image to 3 channels (RGB)."""
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img