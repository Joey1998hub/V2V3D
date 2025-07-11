# V2V3D: View-to-View Denoised 3D Reconstruction for Light-Field Microscopy
This is the official repository for our paper: "V2V3D: View-to-View Denoised 3D Reconstruction for Light-Field Microscopy."

Jiayin Zhao*, Zhenqi Fu*, Tao Yu, Hui Qiao

Tsinghua University & Shanghai AI Laboratory

<div align="center">
  <img src="figures/model.jpg"/>
</div>

#### Train:
* Set the hyper-parameters in `./Config/train.yaml` if needed. We have provided our default settings in the realeased codes.
* Place the input LFs folder into `./Dataset` and the PSFs into `./PSF/` .
* Run `train_model.py` to perform network training.
* Checkpoint will be saved to `./Checkpoints/`.

#### Test:
* Set the hyper-parameters in `./Config/test.yaml` if needed. We have provided our default settings in the realeased codes.
* Place the input LFs folder into `./Dataset` .
* If you want to directly test on the provided LF, download the model via: https://drive.google.com/drive/folders/1UiFG8ChsjGmIZcVV4jzQd0aIcnIvvuEc?usp=sharing
* Run `test_model.py` to perform inference on LFs selected in `dataset.py`.
* The result files will be saved to `./Results`.
  
#### Dataset:
* We have open-sourced the relevant dataset. For details, please refer to: https://doi.org/10.57760/sciencedb.27695
