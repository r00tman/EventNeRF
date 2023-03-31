#!/usr/bin/env bash
# for scene in chair drums ficus hotdog lego materials mic; do
for scene in materials; do
    cd "$scene"

    echo "$scene"
    echo psnr
    ../../main.py event gt | tee event.log
    ../../main.py ngp gt | tee ngp.log
    ../../main.py ngp_log010 gt | tee ngp_log010.log

    echo ssim
    ../../compssim.py event_corr gt | tee event.ssim
    ../../compssim.py ngp_corr gt | tee ngp.ssim
    ../../compssim.py ngp_log010_corr gt | tee ngp_log010.ssim

    echo lpips
    ../../complpips.py event_corr gt | tee event.lpips.alex
    ../../complpips.py ngp_corr gt | tee ngp.lpips.alex
    ../../complpips.py ngp_log010_corr gt | tee ngp_log010.lpips.alex

    cd ..
done

