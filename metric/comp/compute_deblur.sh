#!/usr/bin/env bash
for scene in drums lego; do
    cd "$scene"

    echo "$scene"
    echo psnr
    ../../main.py event gt | tee event.log
    ../../main.py deblur_shifted gt | tee deblur_shifted.log
    ../../main.py deblur_raw gt | tee deblur_raw.log

    echo ssim
    ../../compssim.py event_corr gt | tee event.ssim
    ../../compssim.py deblur_shifted gt | tee deblur_shifted.ssim
    ../../compssim.py deblur_raw gt | tee deblur_raw.ssim

    echo lpips
    ../../complpips.py event_corr gt | tee event.lpips.alex
    ../../complpips.py deblur_shifted gt | tee deblur_shifted.lpips.alex
    ../../complpips.py deblur_raw gt | tee deblur_raw.lpips.alex

    cd ..
done

