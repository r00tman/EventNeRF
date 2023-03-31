#!/usr/bin/env bash
# for scene in chair drums ficus hotdog lego materials mic; do
for scene in materials; do
    cd "$scene"

    echo "$scene"
    echo psnr
    ../../main.py event gt | tee event.log
    ../../main.py e2vid gt | tee e2vid.log

    echo ssim
    ../../compssim.py event_corr gt | tee event.ssim
    ../../compssim.py e2vid_corr gt | tee e2vid.ssim

    echo lpips
    ../../complpips.py event_corr gt | tee event.lpips.alex
    ../../complpips.py e2vid_corr gt | tee e2vid.lpips.alex

    cd ..
done

