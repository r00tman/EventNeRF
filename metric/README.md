Compute PSNR, SSIM, LPIPS metrics for the predictions
---

## Preparation
Please download `comp_data.zip` from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/xDqwRHiWKeSRyes).

Unzip the scene folders with ground-truth, processed data and etc:
```
cd comp
unzip ./comp_data.zip
```

Install the dependencies which may have not been installed:
```
pip install lpips Pillow scikit-image
```


## Automatic operation
All scripts in this section just do steps described in the 'Manual operation' section, but for all of the computed data.

To compute the E2VID comparison numbers reported in the paper run:
```
cd comp
./compute.sh
```

This will color-correct all data and report the metrics to the stardard output and corresponding files, e.g., `comp/mic/event.ssim`.
Note that some of the scenes are commented out in the script.
This is to test if everything works fine on one of the scenes without waiting for everything to finish.

Similarly, to compute NGP numbers and Deblur-NeRF numbers:
```
cd comp
./compute_ngp.sh
./compute_deblur.sh
```

We also provide a script to compose a LaTeX table with all the results:
```
cd comp
./compose_table_h_ngp.py
```

In the next section, we describe how to compute all the numbers manually step-by-step, which is useful to understanding the process and to adapting the workflow to new experiments.


## Manual operation
The first step is to color-correct images and compute PSNR with main.py:
```
./main.py comp/mic/event comp/mic/gt
```

This will output slope & offset of the correction, report PSNR values (only the last one matters and is reported in the paper) and save the results to a folder which is `comp/mic/event_corr` in this example.

Then we compute SSIM and LPIPS:
```
./compssim.py comp/mic/event_corr comp/mic/gt
./complpips.py comp/mic/event_corr comp/mic/gt
```

Note that the corrected `event_corr` is now used, not the original `event`.

## How original files were obtained
Files in `comp/*/event` folder are the results of rendering the learned model.

For example, with the `mic` scene:
```
python ddp_test_nerf.py --config configs/nerf/mic.txt --render_split train --testskip 10
```

These parameters produce the turnaround images that correspond exactly to the ground-truth views we use for evaluation.

It will render and save the images to the corresponding log folder, e.g., `logs/nerf_mic/render_train_660000`.

We only use the final RGB images which have file names in the form `r_*.png`:
```
cp r_*.png comp/mic/event
```
