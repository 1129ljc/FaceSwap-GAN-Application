import os
import cv2
import glob
import time
import numpy as np
from pathlib import PurePath, Path
# from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from networks.faceswap_gan_model import FaceswapGANModel
from keras_vggface.vggface import VGGFace
from data_loader.data_loader import DataLoader
from utils import showG, showG_mask, showG_eyes
import argparse


args_parser = argparse.ArgumentParser()
args_parser.add_argument("--input-faceA-dir", type=str, required=True, help="Directory of extracted faces A.")
args_parser.add_argument("--input-faceB-dir", type=str, required=True, help="Directory of extracted faces B.")
args_parser.add_argument("--input-maskA-dir", type=str, required=True, help="Directory of extracted faces mask A.")
args_parser.add_argument("--input-maskB-dir", type=str, required=True, help="Directory of extracted faces mask B.")
args_parser.add_argument("--face-size", type=int, required=False,  default=0, choices=[64,128,256],
                         help="Size of input faces for network. Choose from [64,128,256]. Default as 64")
args_parser.add_argument("-b","--batch-size", type=int, required=False, default=8, help="Batch size of network. It must be an even number. Default as 8")
args_parser.add_argument("-g","--gpu-num", type=str, required=True, help="numbers of GPU.")
args_parser.add_argument("-i","--iters", type=int, required=False, default=40000, help="numbers of training iters.Default as 40000.")
args_parser.add_argument("--model-dir", type=str, required=True, help="Directory of saved model weights")
args = args_parser.parse_args()

# Number of CPU cores
num_cpus = os.cpu_count()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

# Input/Output resolution
RESOLUTION = args.face_size  # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

# Batch size
batchSize = args.batch_size
assert (batchSize != 1 and batchSize % 2 == 0), "batchSize should be an even number."

# Use motion blurs (data augmentation)
# set True if training data contains images extracted from videos
use_da_motion_blur = False
# 待验证效果

# Use eye-aware training
# require images generated from prep_binary_masks.ipynb
use_bm_eyes = True

# Probability of random color matching (data augmentation)
prob_random_color_match = 0.5

da_config = {
    "prob_random_color_match": prob_random_color_match,
    "use_da_motion_blur": use_da_motion_blur,
    "use_bm_eyes": use_bm_eyes
}

# Path to training images
img_dirA = os.path.join(args.input_faceA_dir,'aligned_faces')
img_dirB = os.path.join(args.input_faceB_dir,'aligned_faces')
img_dirA_bm_eyes = args.input_maskA_dir
img_dirB_bm_eyes = args.input_maskB_dir

# Path to saved model weights
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
models_dir = args.model_dir

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm"  # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard"  # standard, lite

# Loss function weights configuration
loss_weights = {}
loss_weights['w_D'] = 0.1  # Discriminator
loss_weights['w_recon'] = 1.  # L1 reconstruction loss
loss_weights['w_edge'] = 0.1  # edge loss
loss_weights['w_eyes'] = 30.  # reconstruction and edge loss on eyes area
loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1)  # perceptual loss (0.003, 0.03, 0.3, 0.3)

# Init. loss config.
loss_config = {}
loss_config["gan_training"] = "mixup_LSGAN"  # "mixup_LSGAN" or "relativistic_avg_LSGAN"
loss_config['use_PL'] = False
loss_config["PL_before_activ"] = False
loss_config['use_mask_hinge_loss'] = False
loss_config['m_mask'] = 0.
loss_config['lr_factor'] = 1.
loss_config['use_cyclic_loss'] = False

model = FaceswapGANModel(**arch_config)
model.load_weights(path=models_dir)
# VGGFace ResNet50
vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
# vggface.summary()
model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
model.build_train_functions(loss_weights=loss_weights, **loss_config)

train_A = glob.glob(img_dirA + "/*.*")
train_B = glob.glob(img_dirB + "/*.*")

train_AnB = train_A + train_B

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)
print("facesA: " + img_dirA + ' , ' + "masksA: " + img_dirA_bm_eyes)
print("facesB: " + img_dirB + ' , ' + "masksB: " + img_dirB_bm_eyes)
print("Number of images in folder A: " + str(len(train_A)))
print("Number of images in folder B: " + str(len(train_B)))

if use_bm_eyes:
    assert len(glob.glob(img_dirA_bm_eyes + "/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
    assert len(glob.glob(img_dirB_bm_eyes + "/*.*")), "No binary mask found in " + str(img_dirB_bm_eyes)
    assert len(glob.glob(img_dirA_bm_eyes + "/*.*")) == len(train_A), \
        "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder."
    assert len(glob.glob(img_dirB_bm_eyes + "/*.*")) == len(train_B), \
        "Number of faceB images does not match number of their binary masks. Can be caused by any none image file in the folder."


def show_loss_config(loss_config):
    for config, value in loss_config.items():
        print(f"{config} = {value}")


train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
_, tA, bmA = train_batchA.get_next_batch()
_, tB, bmB = train_batchB.get_next_batch()
showG_eyes(tA, tB, bmA, bmB, batchSize)
del train_batchA, train_batchB


def reset_session(save_path):
    global model, vggface
    global train_batchA, train_batchB
    model.save_weights(path=save_path)
    del model
    del vggface
    del train_batchA
    del train_batchB
    K.clear_session()
    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=save_path)
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)


# Start training
t0 = time.time()
'''
# This try/except is meant to resume training that was accidentally interrupted
try:
    gen_iterations
    print(f"Resume training from iter {gen_iterations}.")
except:
    gen_iterations = 0
'''
gen_iterations = 0
errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
errGAs = {}
errGBs = {}
# Dictionaries are ordered in Python 3.6
for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
    errGAs[k] = 0
    errGBs[k] = 0

display_iters = 300
backup_iters = 5000
TOTAL_ITERS = args.iters

train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)

while gen_iterations <= TOTAL_ITERS:

    # Loss function automation
    if gen_iterations == (TOTAL_ITERS // 5 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.0
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (TOTAL_ITERS // 5 + TOTAL_ITERS // 10 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.5
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Complete.")


    elif gen_iterations == (2 * TOTAL_ITERS // 5 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.2
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")

    elif gen_iterations == (TOTAL_ITERS // 2 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.4
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (2 * TOTAL_ITERS // 3 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.
        loss_config['lr_factor'] = 0.3
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")

    elif gen_iterations == (8 * TOTAL_ITERS // 10 - display_iters // 2):
        # print(gen_iterations)
        # clear_output()
        # model.decoder_A.load_weights("models/decoder_B.h5") # swap decoders
        # model.decoder_B.load_weights("models/decoder_A.h5") # swap decoders
        model.decoder_A.load_weights(models_dir + "/decoder_B.h5")  # swap decoders
        model.decoder_B.load_weights(models_dir + "/decoder_A.h5")  # swap decoders
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.1
        loss_config['lr_factor'] = 0.3
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")


    elif gen_iterations == (9 * TOTAL_ITERS // 10 - display_iters // 2):

        # print(gen_iterations)
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.0
        loss_config['lr_factor'] = 0.1
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")

    if gen_iterations == 5:
        print("working.")

    # Train dicriminators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
    errDA_sum += errDA[0]
    errDB_sum += errDB[0]

    # Train generators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
        errGAs[k] += errGA[i]
        errGBs[k] += errGB[i]
    gen_iterations += 1

    # Visualization
    if gen_iterations % display_iters == 0:
        #clear_output()

        # Display loss information
        show_loss_config(loss_config)
        print("----------")
        print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
              % (gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters,
                 errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
        print("----------")
        print("Generator loss details:")
        print(f'[Adversarial loss]')
        print(f'GA: {errGAs["adv"] / display_iters:.4f} GB: {errGBs["adv"] / display_iters:.4f}')
        print(f'[Reconstruction loss]')
        print(f'GA: {errGAs["recon"] / display_iters:.4f} GB: {errGBs["recon"] / display_iters:.4f}')
        print(f'[Edge loss]')
        print(f'GA: {errGAs["edge"] / display_iters:.4f} GB: {errGBs["edge"] / display_iters:.4f}')
        if loss_config['use_PL'] == True:
            print(f'[Perceptual loss]')
            try:
                print(f'GA: {errGAs["pl"][0] / display_iters:.4f} GB: {errGBs["pl"][0] / display_iters:.4f}')
            except:
                print(f'GA: {errGAs["pl"] / display_iters:.4f} GB: {errGBs["pl"] / display_iters:.4f}')

        # Display images
        print("----------")
        wA, tA, _ = train_batchA.get_next_batch()
        wB, tB, _ = train_batchB.get_next_batch()
        print("Transformed (masked) results:")
        showG(tA, tB, model.path_A, model.path_B, batchSize)
        print("Masks:")
        showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)
        print("Reconstruction results:")
        showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)


        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
            errGAs[k] = 0
            errGBs[k] = 0

        # Save models
        model.save_weights(path=models_dir)


'''
# Display random results
wA, tA, _ = train_batchA.get_next_batch()
wB, tB, _ = train_batchB.get_next_batch()
print("Transformed (masked) results:")
showG(tA, tB, model.path_A, model.path_B, batchSize)   
print("Masks:")
showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)  
print("Reconstruction results:")
showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize) 
'''

# (Optional) Additional 40k iterations of training

'''
loss_config['use_PL'] = True
loss_config['use_mask_hinge_loss'] = False
loss_config['m_mask'] = 0.0
loss_config['lr_factor'] = 0.1
reset_session(models_dir)
print("Building new loss funcitons...")
show_loss_config(loss_config)
model.build_train_functions(loss_weights=loss_weights, **loss_config)
'''

"""
# Start training
t0 = time.time()
gen_iterations = 0
errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
errGAs = {}
errGBs = {}
# Dictionaries are ordered in Python 3.6
for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
    errGAs[k] = 0
    errGBs[k] = 0

display_iters = 300
backup_iters = 5000
TOTAL_ITERS = 40000

global train_batchA, train_batchB
train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes, 
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes, 
                          RESOLUTION, num_cpus, K.get_session(), **da_config)

while gen_iterations <= TOTAL_ITERS: 

    if gen_iterations == 5:
        print ("working.")

    # Train dicriminators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
    errDA_sum +=errDA[0]
    errDB_sum +=errDB[0]

    # Train generators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
        errGAs[k] += errGA[i]
        errGBs[k] += errGB[i]
    gen_iterations+=1

    # Visualization
    if gen_iterations % display_iters == 0:
        clear_output()

        # Display loss information
        show_loss_config(loss_config)
        print("----------") 
        print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
        % (gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))  
        print("----------") 
        print("Generator loss details:")
        print(f'[Adversarial loss]')  
        print(f'GA: {errGAs["adv"]/display_iters:.4f} GB: {errGBs["adv"]/display_iters:.4f}')
        print(f'[Reconstruction loss]')
        print(f'GA: {errGAs["recon"]/display_iters:.4f} GB: {errGBs["recon"]/display_iters:.4f}')
        print(f'[Edge loss]')
        print(f'GA: {errGAs["edge"]/display_iters:.4f} GB: {errGBs["edge"]/display_iters:.4f}')
        if loss_config['use_PL'] == True:
            print(f'[Perceptual loss]')
            try:
                print(f'GA: {errGAs["pl"][0]/display_iters:.4f} GB: {errGBs["pl"][0]/display_iters:.4f}')
            except:
                print(f'GA: {errGAs["pl"]/display_iters:.4f} GB: {errGBs["pl"]/display_iters:.4f}')

        # Display images
        print("----------") 
        wA, tA, _ = train_batchA.get_next_batch()
        wB, tB, _ = train_batchB.get_next_batch()
        print("Transformed (masked) results:")
        showG(tA, tB, model.path_A, model.path_B, batchSize)   
        print("Masks:")
        showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)  
        print("Reconstruction results:")
        showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)           
        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
            errGAs[k] = 0
            errGBs[k] = 0

        # Save models
        model.save_weights(path=models_dir)

    # Backup models
    if gen_iterations % backup_iters == 0: 
        bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
        Path(bkup_dir).mkdir(parents=True, exist_ok=True)
        model.save_weights(path=bkup_dir)
"""






