# V2V3D: View-to-View Denoised 3D Reconstruction for Light-Field Microscopy
This is the official repository for our paper: "V2V3D: View-to-View Denoised 3D Reconstruction for Light-Field Microscopy."

Jiayin Zhao*, Zhenqi Fu*, Tao Yu, Hui Qiao

Tsinghua University & Shanghai AI Laboratory

<div align="center">
  <img src="figures/model.jpg"/>
</div>

#### Train:
* Set the hyper-parameters in `./Config/train.yaml` if needed. We have provided our default settings in the realeased codes.
* Run `train_model.py` to perform network training.
* Checkpoint will be saved to `./Checkpoints/`.
* If you want to train the network with your LFM data, place the input LFs into `./Dataset/` and the PSFs into `./PSF/` .

#### Test:
* Place the input LFs folder into `./Dataset` .
* If you want to use our pre-trained model for testing on the provided LF, download the model via: https://drive.google.com/drive/folders/1UiFG8ChsjGmIZcVV4jzQd0aIcnIvvuEc?usp=sharing
* Run `test_model.py` to perform inference on LFs selected in `dataset.py`.
* The result files will be saved to `./Results`.

#### Notes:
* We are organizing the relevant dataset and will release it as soon as possible.
* We will continue to update and improve this repository.
