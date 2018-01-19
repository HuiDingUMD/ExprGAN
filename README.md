# ExprGAN
This is our Tensorflow implementation for our AAAI 2018 oral paper: "ExprGAN: Facial Expression Editing with Controllable Expression Intensity"

# Train
1. Download OULU-CASIA dataset and put the images under data folder: http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA. oulu_anno.pickle contains the split of training and testing images.
2. Download vgg-face.mat from matconvenet website and put is under utils folder:  http://www.vlfeat.org/matconvnet/pretrained/ 
3. Run run_oulu.sh, which trains the model in three stages

A trained model can be downloaded here: https://drive.google.com/open?id=1bz45QSdS2911-8FDmngGIyd5K4gYimzg

# Test
1. Run test_oulu.sh

# Citation
If you use this code for your research, please cite our paper:

@article{ding2017exprgan,
  title={Exprgan: Facial expression editing with controllable expression intensity},
  author={Ding, Hui and Sricharan, Kumar and Chellappa, Rama},
  journal={AAAI},
  year={2018}
}
