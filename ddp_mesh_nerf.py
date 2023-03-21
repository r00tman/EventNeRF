#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
# import mcubes
import marching_cubes as mcubes
import logging
from tqdm import tqdm, trange
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, create_nerf
from nerf_sample_ray_split import CameraManager

logger = logging.getLogger(__package__)

def ddp_mesh_nerf(rank, args):
    ###### set up multi-processing
    assert(args.world_size==1)
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(rank, args, camera_mgr, False)

    # center on lk
    ax = np.linspace(-1, 1, num=300, endpoint=True, dtype=np.float32)
    # X, Y, Z = np.meshgrid(ax, ax, ax+0.4)
    X, Y, Z = np.meshgrid(ax, ax, ax)

    # flip yz
    pts = np.stack((X, Y[::-1], Z[::-1]), -1)/4
    pts = pts.reshape((-1, 3))

    pts = torch.tensor(pts).float().to(rank)

    u = models['net_1']
    nerf_net = u.module.nerf_net
    fg_net = nerf_net.fg_net

    allres = []
    allcolor = []
    with autocast():
        with torch.no_grad():
            # direction = torch.tensor([0, 0, -1], dtype=torch.float32).to(rank)
            for bid in trange((pts.shape[0]+args.chunk_size-1)//args.chunk_size):
                bstart = bid * args.chunk_size
                bend = bstart + args.chunk_size
                cpts = pts[bstart:bend]
                cvd = cpts*0#+direction

                out = fg_net(cpts, cvd, iteration=start,
                             embedder_position=nerf_net.fg_embedder_position,
                             embedder_viewdir=nerf_net.fg_embedder_viewdir)

                res = out['sigma'].detach().cpu().numpy()
                allres.append(res)
                color = out['rgb'].detach().cpu().numpy()
                allcolor.append(color)

    allres = np.concatenate(allres, 0)
    allres = allres.reshape(X.shape)

    allcolor = np.concatenate(allcolor, 0)
    allcolor = allcolor.reshape(list(X.shape)+[3,])

    # print(allres.min(), allres.max(), allres.mean(), np.median(allres), allres.shape)

    logger.info('Doing MC')
    # vtx, tri = mcubes.marching_cubes(allres.astype(np.float32), 100)
    THR=30
    vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), THR)
    logger.info('Exporting mesh')
    # mcubes.export_mesh(vtx, tri, "mesh5.dae", "Mesh")
    mcubes.export_obj(vtx, tri, f"colornet01_scale4_{THR}.obj")


def mesh():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_mesh_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    mesh()

