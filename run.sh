export CUDA_VISIBLE_DEVICES=0
#====================================================== rafd dataset
########### class
RAFD_DATASET_ROOT_DIR=/home/zili/Documents/data/ganimation3/RAFD/rafd512
RAFD_TEST_CKPT_DIR=./pretrained_ckpts/rafd/class
RAFD_TEST_IMG_DIR=./samples/rafd/class/imgs
RAFD_TEST_SIZE=512

#train
#python main.py --control_signal_type class --ngf 64 --aus_nc 8 --coef_flow 0.2 --data_root $RAFD_DATASET_ROOT_DIR --visdom_env rafd --niter 200 --niter_decay 100

#test
python main.py --mode test --control_signal_type class --ngf 64 --aus_nc 8 --coef_flow 0.2 --data_root $RAFD_DATASET_ROOT_DIR --ckpt_dir $RAFD_TEST_CKPT_DIR  --load_epoch 300  --test_dir $RAFD_TEST_IMG_DIR --test_size $RAFD_TEST_SIZE  --save_test_gif  --use_multiscale 


#================================================= emotionnet dataset
########### class
EMOTIONNET_DATASET_ROOT_DIR=/home/zili/Documents/data/ganimation3/emotionnet/
EMOTIONNET_TEST_CKPT_DIR=./pretrained_ckpts/emotionnet/class
EMOTIONNET_TEST_IMG_DIR=./samples/emotionnet/class/imgs
EMOTIONNET_TEST_SIZE=512

#train
##python main.py --control_signal_type class --aus_nc  26 --ngf 64 --coef_flow 0.1 --data_root $EMOTIONNET_DATASET_ROOT_DIR --visdom_env emotionnet --load_size 128  --final_size 128 --niter  200   --niter_decay 100

#test
python main.py --mode test --control_signal_type class --ngf 64 --aus_nc  26 --coef_flow 0.1  --data_root $EMOTIONNET_DATASET_ROOT_DIR --ckpt_dir $EMOTIONNET_TEST_CKPT_DIR --load_epoch 300  --test_dir $EMOTIONNET_TEST_IMG_DIR --test_size $EMOTIONNET_TEST_SIZE --use_multiscale  --save_test_gif


#====================================================== celeba dataset
CELEB_DATASET_ROOT_DIR=/home/zili/Documents/data/ganimation3/celeba

############# AU
DRIVING_SIGNAL_TYPE=au
CELEB_TEST_CKPT_DIR=./pretrained_ckpts/celeba/au/
CELEB_TEST_IMG_DIR=./samples/celeba/au/imgs
CELEB_TEST_AU_DIR=./samples/celeba/au/aus
CELEB_TEST_SIZE=1024

#train
python main.py --control_signal_type $DRIVING_SIGNAL_TYPE --ngf 128 --aus_nc 17 --coef_flow 0.3 --data_root $CELEB_DATASET_ROOT_DIR --visdom_env celeba_au --niter 20 --niter_decay 10

#test
python main.py --mode test --control_signal_type $DRIVING_SIGNAL_TYPE --ngf 128 --aus_nc 17 --coef_flow 0.3 --data_root $CELEB_DATASET_ROOT_DIR --ckpt_dir $CELEB_TEST_CKPT_DIR --load_epoch 30 --test_dir $CELEB_TEST_IMG_DIR  --test_size $CELEB_TEST_SIZE --test_example_dir  $CELEB_TEST_AU_DIR --test_example_type sequence  --save_test_gif --test_example_cropped --use_multiscale 


############# class

DRIVING_SIGNAL_TYPE=class
CELEB_TEST_CKPT_DIR=./pretrained_ckpts/celeba/class
CELEB_TEST_IMG_DIR=./samples/celeba/class/imgs
CELEB_TEST_SIZE=512

#train
#python main.py --control_signal_type $CELEB_TEST_CKPT_DIR --ngf 64 --aus_nc 2 --coef_flow 0.1 --data_root $CELEB_DATASET_ROOT_DIR --visdom_env celeba_class --niter 30 --niter_decay 20

#test
python main.py --mode test --control_signal_type class --ngf 64 --aus_nc 2 --coef_flow 0.1 --data_root $CELEB_DATASET_ROOT_DIR --ckpt_dir $CELEB_TEST_CKPT_DIR --load_epoch 50 --test_dir $CELEB_TEST_IMG_DIR --test_size $CELEB_TEST_SIZE  --use_multiscale --save_test_gif --nframes 10 


