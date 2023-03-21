import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleEventStream
import glob

logger = logging.getLogger(__package__)

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def load_event_data_split(basedir, scene, split, camera_mgr, skip=1, max_winsize=1, use_ray_jitter=True, is_colored=False, polarity_offset=0.0, cycle=False, is_rgb_only=False, randomize_winlen=True, win_constant_count=0):

    def parse_txt(filename, shape):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape(shape).astype(np.float32)

    if basedir[-1] == '/':          # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # event file
    event_file = find_files('{}/events'.format(split_dir), exts=['*.npz'])
    print(event_file)
    assert(len(event_file) == 1)
    event_file = event_file[0]
    event_data = np.load(event_file)
    xs, ys, ts, ps = event_data['x'], event_data['y'], event_data['t'], event_data['p']

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg', '*.JPG', '*.PNG'])
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg', '*.JPG', '*.PNG'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert(len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    for i in range(cam_cnt):
        curr_file = img_files[i]
        if not camera_mgr.contains(curr_file):
            pose = parse_txt(pose_files[i], (4,4))
            camera_mgr.add_camera(curr_file, pose)


    # create ray samplers
    ray_samplers = []
    # 1 for the initial event batch spoiling everything
    # max_winsize more for previous pose not getting into this trap too
    start_range = 0 if cycle else 1+max_winsize
    for i in range(start_range, cam_cnt):
        try:
            intrinsics = parse_txt(intrinsics_files[i], (5,4))
        except ValueError:
            intrinsics = parse_txt(intrinsics_files[i], (4,4))
            # concat unity distortion coefficients
            intrinsics = np.concatenate((intrinsics, np.zeros((1,4), dtype=np.float32)), 0)

        if randomize_winlen:
            winsize = np.random.randint(1, max_winsize+1)
        else:
            winsize = max_winsize


        # what is -1 for? for i=last frame covering all events
        start_time = (i-winsize)/(cam_cnt-1)
        if start_time < 0:
            start_time += 1
        end_time = (i)/(cam_cnt-1)

        end = np.searchsorted(ts, end_time*ts.max())

        if win_constant_count != 0:
            # TODO: there could be a bug with windows in the start, e.g., end-win_constant_count<0
            #       please, check if the windows are correctly composed in that case
            start_time = ts[end-win_constant_count]/ts.max()

            if win_constant_count > end:
                start_time = start_time - 1

            winsize = int(i-start_time*(cam_cnt-1))
            assert(winsize>0)
            start_time = (i-winsize)/(cam_cnt-1)

            if start_time < 0:
                start_time += 1

        start = np.searchsorted(ts, start_time*ts.max())

        if start <= end:
            # normal case: take the interval between
            events = (xs[start:end], ys[start:end], ts[start:end], ps[start:end])
        else:
            # loop over case: compose tail with head events
            events = (np.concatenate((xs[start:], xs[:end])),
                      np.concatenate((ys[start:], ys[:end])),
                      np.concatenate((ts[start:], ts[:end])),
                      np.concatenate((ps[start:], ps[:end])),
                     )

        H, W = 260, 346

        prev_file = img_files[(i-winsize+len(img_files))%len(img_files)]
        curr_file = img_files[i]
        curr_mask = mask_files[i]

        if win_constant_count != 0:
            print('cnt:', len(events[0]), 'request:', win_constant_count)
        ray_samplers.append(RaySamplerSingleEventStream(H=H, W=W, intrinsics=intrinsics,
                                                        events=events,
                                                        rgb_path=curr_file,
                                                        prev_rgb_path=prev_file,
                                                        mask_path=curr_mask,
                                                        end_idx=i,
                                                        use_ray_jitter=use_ray_jitter,
                                                        is_colored=is_colored,
                                                        polarity_offset=polarity_offset,
                                                        is_rgb_only=is_rgb_only))

    logger.info('Split {}, # views: {}, # effective views: {}'.format(split, cam_cnt, len(ray_samplers)))

    return ray_samplers
