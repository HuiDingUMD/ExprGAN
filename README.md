# ExprGAN
This is our Tensorflow implementation for our AAAI 2018 oral paper: "ExprGAN: Facial Expression Editing with Controllable Expression Intensity", https://arxiv.org/pdf/1709.03842.pdf

![Alt text](image/exprgan.png?raw=true "Optional Title")

# Train
1. Download OULU-CASIA dataset and put the images under data folder: http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA. split/oulu_anno.pickle contains the split of training and testing images.
2. Download vgg-face.mat from matconvenet website and put it under joint-train/utils folder:  http://www.vlfeat.org/matconvnet/pretrained/ 
3. To overcome the limited training dataset, the training is consisted of three stages: 
  a) Go inot train-controller folder to first train the controller module, run_controller.sh;
  b) Go into join-train folder for the second and third stage training, run_oulu.sh.
  Plese see our paper for more training details.

A trained model can be downloaded here: https://drive.google.com/open?id=1bz45QSdS2911-8FDmngGIyd5K4gYimzg

# Test
1. Run joint-train/test_oulu.sh

# Citation
If you use this code for your research, please cite our paper:

@article{ding2017exprgan,
  title={Exprgan: Facial expression editing with controllable expression intensity},
  author={Ding, Hui and Sricharan, Kumar and Chellappa, Rama},
  journal={AAAI},
  year={2018}
}
