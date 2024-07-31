# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import random
import argparse
import warnings
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as ttF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Lambda
from torchvision import datasets, transforms

import utils


def _tuple2(x):
    assert isinstance(x, (int, float))
    return (x,x)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if ttF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))
        
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            if 'RandomCropCoord' in t.__class__.__name__:
                img, kwargs['coord'] = t(img, **kwargs)
            elif 'FlipCoord' in t.__class__.__name__:
                assert 'coord' in kwargs
                img, kwargs['coord'] = t(img, kwargs['coord'])
            else:
                img = t(img)
        return img, kwargs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p: #'ij' coords
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return ttF.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:  #'ij' coords
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return ttF.vflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCropCoord(object):
    """Crop the given PIL Image to a random size and aspect ratio ** without resize

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. 
    
    Args:
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
            
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            #take log to even the distribution |log(10)-log(0)| = |log(0)-log(10)|
            log_ratio = (math.log(ratio[0]), math.log(ratio[1])) 
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                #random upper left corner
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    @staticmethod
    def get_params_ref(img, scale, ratio, ref_crop):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        random_crop = False
        width, height = _get_image_size(img)
        area = height * width
        
        ri, rj, rI, rJ = ref_crop.tolist()
        # unflip the coordinates:
        if ri > rI:
            ri, rI = rI, ri
        if rj > rJ:
            rj, rJ = rJ, rj
        ri, rj, rI, rJ = round(ri * (height-1)), round(rj * (width-1)), round(rI * (height-1)), round(rJ * (width-1))
        rh, rw = rI - ri, rJ - rj
        ri_c, rj_c = (rI + ri) / 2, (rJ + rj) / 2 

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            #take log to even the distribution |log(10)-log(0)| = |log(0)-log(10)|
            log_ratio = (math.log(ratio[0]), math.log(ratio[1])) 
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                #random upper left corner
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                random_crop = True
                break

        # Fallback to central crop
        if not random_crop:
            in_ratio = float(width) / float(height)
            if (in_ratio < min(ratio)):
                w = width
                h = int(round(w / min(ratio)))
            elif (in_ratio > max(ratio)):
                h = height
                w = int(round(h * max(ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        I, J = i + h - 1, j + w - 1
        i_c, j_c = (i + I) / 2, (j + J) / 2
        bb_h, bb_w = max(I, rI) - min(i, ri), max(J, rJ) - min(i, rj)
        
        #non-overlap crops:
        i_shift = bb_h - h - rh if bb_h > h + rh else 0
        j_shift = bb_w - w - rw if bb_w > w + rw else 0
        
        #make sure >= 0.5**2 = 25% overlap
        h_min, w_min = min(rh, h), min(rw, w)
        i_shift += 0.5 * h_min if bb_h >= h + rh - 0.5 * h_min else 0
        j_shift += 0.5 * w_min if bb_w >= w + rw - 0.5 * w_min else 0

        #shift with direction:
        i += round((ri_c - i_c) / (abs(ri_c - i_c) + 1e-5) * i_shift)
        j += round((rj_c - j_c) / (abs(rj_c - j_c) + 1e-5) * j_shift)
        
        #Bound check:
        if i + h > height:
            i = height - h 
        if i < 0:
            i = 0
        if j + w > width:
            j = width - w
        if j < 0:
            j = 0
        #print('h','w', 'ref_crop', 'crop', height, width, '(', ri, rj, rI, rJ, ')', '(',i,j,i+h,j+w,')')
        return i, j, h, w, height, width

    def __call__(self, img, **kwargs):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if 'ref' in kwargs:
            i, j, h, w, height, width = self.get_params_ref(img, self.scale, self.ratio, kwargs['ref']) #ij coords
        else:
            i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio) #ij coords

        # 'ij' coords
        coord = torch.Tensor([float(i) / (height - 1), float(j) / (width - 1), 
                              float(i + h - 1) / (height - 1), float(j + w - 1) / (width - 1)])
        
        return ttF.crop(img, i, j, h, w), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class DataAugmentation(object):
    def __init__(self, 
                 global_crops_scale=(.25, 1.), 
                 local_crops_scale=(.05, .25), 
                 local_crops_number=6,
                 global_size=224,
                 local_size=96,
                 sampling_window_size_g=3,
                 sampling_window_size_l=2,
                 patch_size=16,
                 random_sampling=True,
                 horizontal_flip=0,
                 vertical_flip=0,
                 **kwargs):
        
        self.global_size=global_size
        self.local_size=local_size  
        self.local_crops_number = local_crops_number
        self.global_feature_size = global_size // patch_size
        self.global_num_token = self.global_feature_size ** 2
        self.local_feature_size = local_size //patch_size
        self.local_num_token = self.local_feature_size ** 2
                     
        self.sampling_window_size_g=sampling_window_size_g
        self.sampling_window_size_l=sampling_window_size_l
        self.sampling_num_g = (self.global_feature_size // self.sampling_window_size_g) ** 2
        self.sampling_num_l = (self.local_feature_size // self.sampling_window_size_l) ** 2
        self.random_sampling = random_sampling

        self.global_normalized_meshgrid = torch.stack(torch.meshgrid(torch.linspace(0,1, self.global_feature_size), 
                                                         torch.linspace(0,1, self.global_feature_size), indexing='ij'), 
                                          dim=-1).reshape(-1, 2).unsqueeze(0)

        self.local_normalized_meshgrid = torch.stack(torch.meshgrid(torch.linspace(0,1, self.local_feature_size), 
                                                         torch.linspace(0,1, self.local_feature_size), indexing='ij'), 
                                          dim=-1).reshape(-1, 2).unsqueeze(0)

        
        # color jitter, grayscale
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])

        random_flip = []
        if horizontal_flip > 0:
            random_flip.append(RandomHorizontalFlipCoord(p=horizontal_flip))
        if vertical_flip > 0:
            random_flip.append(RandomVerticalFlipCoord(p=vertical_flip))
                     
        # normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # global crop
        global_resize = transforms.Resize(_tuple2(self.global_size), interpolation=InterpolationMode.BICUBIC)  
        self.global_crop = Compose([RandomCropCoord(scale=global_crops_scale), *random_flip, global_resize])    
                     
        # local crop
        local_resize = transforms.Resize(_tuple2(self.local_size), interpolation=InterpolationMode.BICUBIC) 
        self.local_crop = Compose([RandomCropCoord(scale=local_crops_scale), *random_flip, local_resize])
        
        #transform
        self.transformation = transforms.Compose([color_jitter,
                                                  utils.RandomSelect(
                                                      utils.RandomSelect(
                                                          utils.GaussianBlur(p=0.5),
                                                          utils.GaussianBlur(p=1.0), 
                                                          p=0.5),
                                                      transforms.Compose([
                                                          utils.GaussianBlur(.1),
                                                          utils.Solarization(.2),]),
                                                      p=0.5,),
                                                  self.normalize,])
        
    
    def _sampling_mask(self, feature_size, window_size=2):
        
        grid_size = feature_size // window_size
        sample_feature_size = grid_size * window_size
        window_length = window_size ** 2
        
        selections = torch.randint(0, window_length, (grid_size ** 2,)) if self.random_sampling \
            else torch.zeros(grid_size ** 2, dtype=torch.int64)
        mask = F.one_hot(selections, num_classes=window_length).type(torch.float32)\
                .reshape(-1, grid_size, grid_size, window_size, window_size)\
                .transpose(-2,-3).reshape(sample_feature_size, sample_feature_size)
        if feature_size > sample_feature_size:
            mask_s = mask
            mask = torch.zeros(feature_size, feature_size)
            p = random.random()
            if p < .25:
                mask[:sample_feature_size,:sample_feature_size] = mask_s
            elif p >= .25 and p < .5:
                mask[:sample_feature_size,-sample_feature_size:] = mask_s
            elif p >=.5 and p < .75:
                mask[-sample_feature_size:,:sample_feature_size] = mask_s
            elif p >= .75:
                mask[-sample_feature_size:,-sample_feature_size:] = mask_s
        return (mask>0).flatten(0) # N


    def _project_sampling(self, coord, mask):
        # input, a list of
        # coord: 
        # [i_upleft/(height-1), j_upleft/(width-1),
        #  i_lowright/(height-1), j_lowright/(width-1)]
    
        number_token = mask.shape[-1] #N
        meshgrid = self.global_normalized_meshgrid if number_token == self.global_num_token \
                   else self.local_normalized_meshgrid
        sampled_coords = coord[:2].reshape(1,1,-1) + \
                         meshgrid[mask.unsqueeze(0)] * (coord[2:] - coord[:2])
        return sampled_coords


    def _unflip_crop_coord(self, coord):
        unflipped_coord = coord.clone()
        if coord[2] - coord[0] < 0:
            unflipped_coord[0], unflipped_coord[2] = coord[2], coord[0]
        if coord[3] - coord[1] < 0:
            unflipped_coord[1], unflipped_coord[3] = coord[3], coord[1]
            
        return unflipped_coord


    def _crop_intersection(self, coord1, coord2):
        ul, _= torch.cat([coord1[None,:2],coord2[None, :2]], dim=0).min(dim=0)
        br, _= torch.cat([coord1[None,2:],coord2[None, 2:]], dim=0).max(dim=0)
        return torch.cat([ul, br])
        

    def _create_map(self, src_crop, tgt_crop, mask):
        tgt_scale = tgt_crop[2:] - tgt_crop[:2]
        num_samples = mask.sum().item()
        src_mask = mask.clone()
        
        src_ij = self._project_sampling(src_crop, src_mask) #[1, 16 or 9, 2]
        tgt_crop_unf = self._unflip_crop_coord(tgt_crop)
        in_mask = (((src_ij - tgt_crop_unf[None, None, :2])>0) * ((tgt_crop_unf[None, None, 2:] - src_ij)>0)).all(dim=-1) #[1, 16 or 9]
        
        src_mask[mask] = in_mask
        num_ij = src_mask.sum().item()
        src_maps = torch.zeros(num_samples, len(src_mask))
        src_maps[:num_ij] = torch.diag(src_mask)[src_mask].float()
       
        src_ij = src_ij[in_mask].squeeze(0) #[n,2]
        tgt_ij = (src_ij - tgt_crop[None, :2]) * (self.global_feature_size - 1) / tgt_scale[None, :]
        
        tgt_maps = torch.zeros(num_samples, self.global_feature_size, self.global_feature_size)
        tgt_i = tgt_ij[:, 0]
        tgt_j = tgt_ij[:, 1]
        tgt_i_f = tgt_i.floor().int()
        tgt_i_fl = tgt_i_f.tolist()
        tgt_j_f = tgt_j.floor().int()
        tgt_j_fl = tgt_j_f.tolist()
        tgt_i_c = tgt_i_f + 1
        tgt_i_c[tgt_i_c==self.global_feature_size] = self.global_feature_size-1
        tgt_i_cl = tgt_i_c.tolist()
        tgt_j_c = tgt_j_f + 1
        tgt_j_c[tgt_j_c==self.global_feature_size] = self.global_feature_size-1
        tgt_j_cl = tgt_j_c.tolist()
        
        mask_id = list(range(num_ij))
        tgt_maps[mask_id, tgt_i_fl, tgt_j_fl] += (tgt_i_c - tgt_i) * (tgt_j_c - tgt_j)
        tgt_maps[mask_id, tgt_i_fl, tgt_j_cl] += (tgt_i_c - tgt_i) * (tgt_j - tgt_j_f)
        tgt_maps[mask_id, tgt_i_cl, tgt_j_fl] += (tgt_i - tgt_i_f) * (tgt_j_c - tgt_j)
        tgt_maps[mask_id, tgt_i_cl, tgt_j_cl] += (tgt_i - tgt_i_f) * (tgt_j - tgt_j_f)
        tgt_maps = tgt_maps.flatten(-2)
        
        return src_maps, tgt_maps
    
    
    def __call__(self, image):
        data = defaultdict(list)
        # student global crop
        crop, coord= self.global_crop(image)
        data['crops'].append(self.transformation(crop))
        data['coords'].append(coord['coord'])

        #teacher global crop
        crop, coord = self.global_crop(image, ref=coord['coord'])
        data['crops'].append(self.transformation(crop))
        data['coords'].append(coord['coord'])

        #data['masks']
        sample= self._sampling_mask(self.global_feature_size, window_size=self.sampling_window_size_g)
        sample_s, map_t = self._create_map(data['coords'][0], data['coords'][1], sample) #student --> teacher
        sample_t, map_s = self._create_map(data['coords'][1], data['coords'][0], sample) #teacher --> student
        
        data['src_maps'] += [sample_s, sample_t]
        data['tgt_maps'] += [map_t, map_s]
        
        crop_isec = self._crop_intersection(data['coords'][0], data['coords'][1])
        
        sample=self._sampling_mask(self.local_feature_size, window_size=self.sampling_window_size_l)
        samples_l2t, samples_l2s = [],[]
    
        for _ in range(self.local_crops_number):
            crop, coord = self.local_crop(image, ref=crop_isec)
            data['crops'].append(self.transformation(crop))
            data['coords'].append(coord['coord'])
            
            #local crop to teacher crop
            sample_l, map_l = self._create_map(coord['coord'], data['coords'][1], sample)
            samples_l2t.append(sample_l)
            data['tgt_maps'].append(map_l)
            
            #local crop to student crop
            sample_l, map_l = self._create_map(coord['coord'], data['coords'][0], sample)
            samples_l2s.append(sample_l)
            data['tgt_maps'].append(map_l)

        data['src_maps'] += samples_l2t + samples_l2s

        del samples_l2t, samples_l2s 
        return data