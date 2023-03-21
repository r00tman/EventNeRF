import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import cv2
import imageio
import numba

import logging
logger = logging.getLogger(__package__)
########################################################################################################################
# ray batch sampling
########################################################################################################################
# def get_rays_single_image(H, W, intrinsics, c2w):
#     '''
#     :param H: image height
#     :param W: image width
#     :param intrinsics: 4 by 4 intrinsic matrix
#     :param c2w: 4 by 4 camera to world extrinsic matrix
#     :return:
#     '''
#     u, v = np.meshgrid(np.arange(W), np.arange(H))

#     u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
#     v = v.reshape(-1).astype(dtype=np.float32) + 0.5
#     pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

#     ray_matrix = np.dot(c2w[:3, :3], np.linalg.inv(intrinsics[:3, :3]))
#     # rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
#     # rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
#     rays_d = np.dot(ray_matrix, pixels)  # (3, H*W)
#     rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

#     rays_o = c2w[:3, 3].reshape((1, 3))
#     rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

#     depth = np.linalg.inv(c2w)[2, 3]
#     depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

#     return rays_o, rays_d, depth, ray_matrix

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''

    intrinsics = torch.from_numpy(intrinsics).to(c2w.device)
    # c2w = torch.from_numpy(c2w)

    u, v = torch.meshgrid(torch.arange(W, device=c2w.device), torch.arange(H, device=c2w.device))
    u, v = u.T, v.T

    u = u.reshape(-1).float() + 0.5    # add half pixel
    v = v.reshape(-1).float() + 0.5

    # ---- distort rays to match the camera ----
    f = [intrinsics[0,0], intrinsics[1,1]]
    c = [intrinsics[0,2], intrinsics[1,2]]
    k = [intrinsics[4,0], intrinsics[4,1]]
    # logger.info(f'{f} {c} {k} {u.max()} {v.max()}')

    x = (u-c[0])/f[0]
    y = (v-c[1])/f[1]
    r2 = x**2+y**2
    dist = (1+k[0]*r2+k[1]*r2*r2)
    x = x/dist
    y = y/dist
    u = x*f[0]+c[0]
    v = y*f[1]+c[1]
    # logger.info(f'{u.min()} {v.min()} {u.max()} {v.max()} {x.min()} {x.max()} {y.min()} {y.max()}')
    # ------------------------------------------


    pixels = torch.stack((u, v, torch.ones_like(u)), axis=0)  # (3, H*W)

    ray_matrix = torch.matmul(c2w[:3, :3], torch.inverse(intrinsics[:3, :3]))
    # rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    # rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = torch.matmul(ray_matrix, pixels)  # (3, H*W)
    rays_d = rays_d.transpose(1, 0)  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = torch.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = torch.inverse(c2w)[2, 3]
    depth = depth * torch.ones((rays_o.shape[0],), dtype=rays_o.dtype, device=rays_o.device)  # (H*W,)

    # rays_o = rays_o.cpu().numpy()
    # rays_d = rays_d.cpu().numpy()
    # depth = depth.cpu().numpy()
    # ray_matrix = ray_matrix.cpu().numpy()

    return rays_o, rays_d, depth, ray_matrix


class CameraManager(nn.Module):
    def __init__(self, learnable=False):
        super().__init__()
        self.learnable = learnable
        self.c2w_store = nn.ParameterDict()

    def encode_name(self, name):
        return name.replace('.', '-')

    def add_camera(self, name, c2w):
        key = self.encode_name(name)
        self.c2w_store[key] = nn.Parameter(torch.from_numpy(c2w))

    def contains(self, name):
        key = self.encode_name(name)
        return key in self.c2w_store

    def get_c2w(self, name):
        key = self.encode_name(name)
        res = self.c2w_store[key]
        if not self.learnable:
            res = res.detach()
        return res


# class RaySamplerSingleImage(object):
#     def __init__(self, H, W, intrinsics, c2w,
#                        img_path=None,
#                        resolution_level=1,
#                        mask_path=None,
#                        min_depth_path=None,
#                        max_depth=None,
#                        use_ray_jitter=True):
#         super().__init__()
#         self.W_orig = W
#         self.H_orig = H
#         self.intrinsics_orig = intrinsics
#         self.c2w_mat = c2w

#         self.img_path = img_path
#         self.mask_path = mask_path
#         self.min_depth_path = min_depth_path
#         self.max_depth = max_depth

#         self.resolution_level = -1
#         self.set_resolution_level(resolution_level)

#         self.use_ray_jitter = use_ray_jitter

#     def set_resolution_level(self, resolution_level):
#         if resolution_level != self.resolution_level:
#             self.resolution_level = resolution_level
#             self.W = self.W_orig // resolution_level
#             self.H = self.H_orig // resolution_level
#             self.intrinsics = np.copy(self.intrinsics_orig)
#             self.intrinsics[:2, :3] /= resolution_level

#             if self.mask_path is not None:
#                 self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
#                 self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
#                 if len(self.mask.shape) == 3:  # if RGB mask, take R
#                     print('mask shape', self.mask.shape, 'taking first channel only')
#                     self.mask = self.mask[..., 0]
#                 self.mask = self.mask.reshape((-1,))
#             else:
#                 self.mask = None
#             # only load image at this time
#             if self.img_path is not None:
#                 self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
#                 self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
#                 if self.img.shape[2] == 4:
#                     self.mask = self.img[..., 3].reshape((-1,))
#                     self.img = self.img[..., :3].reshape((-1, 3))
#                 elif self.img.shape[2] == 3:
#                     self.img = self.img.reshape((-1, 3))
#             else:
#                 self.img = None


#             if self.min_depth_path is not None:
#                 self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
#                 self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
#                 self.min_depth = self.min_depth.reshape((-1))
#             else:
#                 self.min_depth = None

#             self.rays_o, self.rays_d, self.depth, self.ray_matrix = get_rays_single_image(self.H, self.W,
#                                                                                           self.intrinsics, self.c2w_mat)

#     def get_img(self):
#         if self.img is not None:
#             return self.img.reshape((self.H, self.W, 3))
#         else:
#             return None

#     def get_all(self):
#         if self.min_depth is not None:
#             min_depth = self.min_depth
#         else:
#             min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

#         ret = OrderedDict([
#             ('ray_o', self.rays_o),
#             ('ray_d', self.rays_d),
#             ('depth', self.depth),
#             ('rgb', self.img),
#             ('mask', self.mask),
#             ('min_depth', min_depth),
#         ])
#         # return torch tensors
#         for k in ret:
#             if ret[k] is not None:
#                 ret[k] = torch.from_numpy(ret[k])
#         return ret

#     def random_sample(self, N_rand, center_crop=False):
#         '''
#         :param N_rand: number of rays to be casted
#         :return:
#         '''
#         if center_crop:
#             half_H = self.H // 2
#             half_W = self.W // 2
#             quad_H = half_H // 2
#             quad_W = half_W // 2

#             # pixel coordinates
#             u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
#                                np.arange(half_H-quad_H, half_H+quad_H))
#             u = u.reshape(-1)
#             v = v.reshape(-1)

#             select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

#             # Convert back to original image
#             select_inds = v[select_inds] * self.W + u[select_inds]
#         else:
#             # Random from one image
#             select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

#         rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
#         rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
#         depth = self.depth[select_inds]         # [N_rand, ]
#         ray_matrix = self.ray_matrix

#         noise = np.random.rand(2, len(select_inds)).astype(np.float32)-0.5  # [2, N_rand]
#         noise = np.stack((noise[0], noise[1], np.zeros(len(select_inds), dtype=np.float32)), axis=0)  # [3, N_rand]
#         noise = np.dot(ray_matrix, noise)
#         noise = noise.transpose((1, 0))  # [N_rand, 3]
#         assert(noise.shape == rays_d.shape)

#         if self.use_ray_jitter:
#             rays_d = rays_d + noise

#         if self.img is not None:
#             rgb = self.img[select_inds, :]          # [N_rand, 3]
#         else:
#             rgb = None

#         if self.mask is not None:
#             mask = self.mask[select_inds]
#         else:
#             mask = None

#         if self.min_depth is not None:
#             min_depth = self.min_depth[select_inds]
#         else:
#             min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

#         ret = OrderedDict([
#             ('ray_o', rays_o),
#             ('ray_d', rays_d),
#             ('depth', depth),
#             ('rgb', rgb),
#             ('mask', mask),
#             ('min_depth', min_depth),
#             ('img_name', self.img_path)
#         ])
#         # return torch tensors
#         for k in ret:
#             if isinstance(ret[k], np.ndarray):
#                 ret[k] = torch.from_numpy(ret[k])

#         return ret


@numba.jit()
def accumulate_events(xs, ys, ts, ps, out, resolution_level, polarity_offset):
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        out[y // resolution_level, x // resolution_level] += p+polarity_offset


class RaySamplerSingleEventStream(object):
    def __init__(self, H, W, intrinsics,
                       events=None,
                       rgb_path=None,
                       prev_rgb_path=None,
                       mask_path=None,
                       resolution_level=1,
                       end_idx=None,
                       use_ray_jitter=True,
                       is_colored=False,
                       polarity_offset=0.0,
                       is_rgb_only=False):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics

        self.events = events
        self.event_frame = None
        self.is_colored = is_colored
        self.polarity_offset = polarity_offset
        self.is_rgb_only = is_rgb_only

        self.img_path = str(end_idx)

        self.mask_path = mask_path
        self.rgb_path = rgb_path
        self.prev_rgb_path = prev_rgb_path

        self.rgb = None
        self.prev_rgb = None
        self.mask = None

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

        self.use_ray_jitter = use_ray_jitter

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level

            self.event_frame = np.zeros((self.H, self.W))
            if not self.is_rgb_only:
                xs, ys, ts, ps = self.events
                accumulate_events(xs, ys, ts, ps, self.event_frame, resolution_level, self.polarity_offset)
                # for x, y, t, p in zip(*self.events):
                #     self.event_frame[y // resolution_level, x // resolution_level] += p+self.polarity_offset

            self.event_frame = np.tile(self.event_frame[..., None], (1, 1, 3))
            self.event_frame = self.event_frame.reshape((-1, 3))

            self.color_mask = np.zeros((self.H, self.W, 3))

            if self.is_colored:
                self.color_mask[0::2, 0::2, 0] = 1  # r

                self.color_mask[0::2, 1::2, 1] = 1  # g
                self.color_mask[1::2, 0::2, 1] = 1  # g

                self.color_mask[1::2, 1::2, 2] = 1  # b
            else:
                self.color_mask[...] = 1

            self.color_mask = self.color_mask.reshape((-1, 3))

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                if len(self.mask.shape) == 3:  # if RGB mask, take R
                    logger.warning(f'mask shape {self.mask.shape} - taking first channel only')
                    self.mask = self.mask[..., 0]
                self.mask = self.mask.reshape((-1,))
            else:
                self.mask = None

            if self.rgb_path is not None:
                assert(self.prev_rgb_path is not None)
                self.prev_rgb = np.zeros((self.H, self.W), dtype=np.float32)
                # self.prev_rgb = imageio.imread(self.prev_rgb_path).astype(np.float32) / 255.
                # self.prev_rgb = cv2.resize(self.prev_rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
                print(self.rgb_path, self.prev_rgb_path)
                if len(self.prev_rgb.shape) == 2:
                    self.prev_rgb = np.tile(self.prev_rgb[..., None], (1, 1, 3)).reshape((-1, 3))
                elif self.prev_rgb.shape[2] == 4:
                    # self.mask = self.prev_rgb[..., 3].reshape((-1,))
                    self.prev_rgb = self.prev_rgb[..., :3].reshape((-1, 3))
                elif self.prev_rgb.shape[2] == 3:
                    self.prev_rgb = self.prev_rgb.reshape((-1, 3))

                self.rgb = np.zeros((self.H, self.W), dtype=np.float32)
                # self.rgb = imageio.imread(self.rgb_path).astype(np.float32) / 255.
                # self.rgb = cv2.resize(self.rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
                if len(self.rgb.shape) == 2:
                    self.rgb = np.tile(self.rgb[..., None], (1, 1, 3)).reshape((-1, 3))
                elif self.rgb.shape[2] == 4:
                    # self.mask = self.rgb[..., 3].reshape((-1,))
                    self.rgb = self.rgb[..., :3].reshape((-1, 3))
                elif self.rgb.shape[2] == 3:
                    self.rgb = self.rgb.reshape((-1, 3))
            else:
                self.rgb = None
                self.prev_rgb = None

        self.prev_rays_o, self.prev_rays_d, self.prev_depth, self.prev_ray_matrix = None, None, None, None
        self.rays_o, self.rays_d, self.depth, self.ray_matrix = None, None, None, None

    def update_rays(self, camera_mgr):
        prev_c2w_mat = camera_mgr.get_c2w(self.prev_rgb_path)
        c2w_mat = camera_mgr.get_c2w(self.rgb_path)

        self.prev_rays_o, self.prev_rays_d, self.prev_depth, self.prev_ray_matrix = \
                get_rays_single_image(self.H, self.W, self.intrinsics, prev_c2w_mat)

        self.rays_o, self.rays_d, self.depth, self.ray_matrix = \
                get_rays_single_image(self.H, self.W, self.intrinsics, c2w_mat)

    def get_img(self):
        if self.event_frame is not None:
            return self.event_frame.reshape((self.H, self.W, 3))
        else:
            return None

    def get_rgb(self):
        if self.rgb is not None:
            return self.rgb.reshape((self.H, self.W, 3))
        else:
            return None

    def get_all(self):
        min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0])

        ret = OrderedDict([
            ('prev_ray_o', self.prev_rays_o),
            ('prev_ray_d', self.prev_rays_d),
            ('prev_depth', self.prev_depth),

            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('events', self.event_frame),
            ('min_depth', min_depth),

            ('rgb', self.rgb),
            ('prev_rgb', self.prev_rgb),
            ('color_mask', self.color_mask),
            ('mask', self.mask),
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False, neg_ratio=0):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            mask = np.nonzero(self.event_frame[..., 0])
            # print(mask)
            assert(len(mask) == 1)
            mask = mask[0]

            if mask.shape[0] > 0 and not self.is_rgb_only:
                # Random from one image
                # select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
                pos_size = int(N_rand*(1-neg_ratio))
                pos_should_replace = pos_size>mask.shape[0]
                if pos_should_replace:
                    logger.warning('sampling views with replacement (not enough events this frame)')
                select_inds_raw = np.random.choice(mask.shape[0], size=(pos_size,), replace=pos_should_replace)

                select_inds = mask[select_inds_raw]

                neg_inds = np.random.choice(self.H*self.W, size=(N_rand-select_inds_raw.shape[0],), replace=False)
                select_inds = np.concatenate([select_inds, neg_inds])
            else:
                select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
                if not self.is_rgb_only:
                    logger.warning('no events this frame, bad sampling')

        prev_rays_o = self.prev_rays_o[select_inds, :]    # [N_rand, 3]
        prev_rays_d = self.prev_rays_d[select_inds, :]    # [N_rand, 3]
        prev_depth = self.prev_depth[select_inds]         # [N_rand, ]
        prev_ray_matrix = self.prev_ray_matrix

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]
        ray_matrix = self.ray_matrix

        # noise = np.random.rand(2, len(select_inds)).astype(np.float32)-0.5  # [2, N_rand]
        # noise = np.stack((noise[0], noise[1], np.zeros(len(select_inds), dtype=np.float32)), axis=0)  # [3, N_rand]
        # ornoise = noise
        # noise = np.dot(ray_matrix, noise)
        # noise = noise.transpose((1, 0))  # [N_rand, 3]
        #
        # prev_noise = np.dot(prev_ray_matrix, ornoise)
        # prev_noise = prev_noise.transpose((1, 0))  # [N_rand, 3]
        # assert(noise.shape == rays_d.shape)
        # assert(prev_noise.shape == prev_rays_d.shape)
        #
        # if self.use_ray_jitter:
        #     rays_d = rays_d + noise
        #     prev_rays_d = prev_rays_d + prev_noise

        if self.events is not None:
            events = self.event_frame[select_inds, :]          # [N_rand, 3]
        else:
            events = None

        if self.prev_rgb is not None:
            prev_rgb = self.prev_rgb[select_inds, :]          # [N_rand, 3]
        else:
            prev_rgb = None

        if self.rgb is not None:
            rgb = self.rgb[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        min_depth = 1e-4 * torch.ones_like(rays_d[..., 0])

        color_mask = self.color_mask[select_inds, :]

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('prev_ray_o', prev_rays_o),
            ('prev_ray_d', prev_rays_d),
            ('prev_depth', prev_depth),
            ('events', events),
            ('min_depth', min_depth),
            ('img_name', self.img_path),

            ('rgb', rgb),
            ('prev_rgb', prev_rgb),
            ('color_mask', color_mask),
            ('mask', mask),
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret
