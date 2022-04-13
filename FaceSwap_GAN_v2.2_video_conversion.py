import os
import keras.backend as K
from networks.faceswap_gan_model import FaceswapGANModel
from converter.video_converter import VideoConverter
from detector.face_detector import MTCNNFaceDetector
import argparse


args_parser = argparse.ArgumentParser()
args_parser.add_argument("--input", type=str, required=False, help="Directory of extracted source video.")
args_parser.add_argument("--output", type=str, required=True, help="Directory of target video.")
args_parser.add_argument("--face-size", type=int, required=False,  default=0, choices=[64,128,256],
                         help="Size of input faces for network. Choose from [64,128,256]. Default as 64.")
args_parser.add_argument("--output-type", type=int, required=False, default=2, 
                        help="Output type parameters : 1. [ result ], 2. [ source | result ], 3. [ source | result | mask ]. Default as 1.")
args_parser.add_argument("--merge-type", type=str, required=False,  default="AtoB", choices=["AtoB","BtoA"],
                         help="Direction of face transformation. Choose from [ AtoB, BtoA ]. Default as AtoB.")
args_parser.add_argument("-b","--batch-size", type=int, required=False, default=8, help="Batch size of network. It must be an even number. Default as 8.")
args_parser.add_argument("-g","--gpu-num", type=str, required=True, help="numbers of GPU.")
args_parser.add_argument("--model-dir", type=str, required=True, help="Directory of model weights")
args_parser.add_argument("--mode", type=str, required=True,choices=["video","picture"],
                         help="Selection of testing mode.")
args = args_parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
K.set_learning_phase(0)

# Input/Output resolution
RESOLUTION = args.face_size  # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm"  # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard"  # standard, lite

model = FaceswapGANModel(**arch_config)
model.load_weights(path=args.model_dir)

mtcnn_weights_dir = "./mtcnn_weights/"
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
vc = VideoConverter()
vc.set_face_detector(fd)
vc.set_gan_model(model)
'''
Video conversion configuration
use_smoothed_bbox:
Boolean. Whether to enable smoothed bbox.
use_kalman_filter:
Boolean. Whether to enable Kalman filter.
use_auto_downscaling:
Boolean. Whether to enable auto-downscaling in face detection (to prevent OOM error).
bbox_moving_avg_coef:
Float point between 0 and 1. Smoothing coef. used when use_kalman_filter is set False.
min_face_area:
int x int. Minimum size of face. Detected faces smaller than min_face_area will not be transformed.
IMAGE_SHAPE:
Input/Output resolution of the GAN model
kf_noise_coef:
Float point. Increase by 10x if tracking is slow. Decrease by 1/10x if trakcing works fine but jitter occurs.
use_color_correction:
String of "adain", "adain_xyz", "hist_match", or "none". The color correction method to be applied.
detec_threshold:
Float point between 0 and 1. Decrease its value if faces are missed. Increase its value to reduce false positives.
roi_coverage:
Float point between 0 and 1 (exclusive). Center area of input images to be cropped (Suggested range: 0.85 ~ 0.95)
enhance:
Float point. A coef. for contrast enhancement in the region of alpha mask (Suggested range: 0. ~ 0.4)
output_type:
Layout format of output video: 1. [ result ], 2. [ source | result ], 3. [ source | result | mask ]
direction:
String of "AtoB" or "BtoA". Direction of face transformation.
'''

options = {
    # ===== Fixed =====
    "use_smoothed_bbox": True,
    "use_kalman_filter": True,
    "use_auto_downscaling": True,
    "bbox_moving_avg_coef": 0.65,
    "min_face_area": 35 * 35,
    "IMAGE_SHAPE": model.IMAGE_SHAPE,
    # ===== Tunable =====
    "kf_noise_coef": 3e-3,
    "use_color_correction": "hist_match",
    "detec_threshold": 0.7,
    "roi_coverage": 0.9,
    "enhance": 0.,
    # "output_type": 3,
    "output_type": args.output_type,
    # important parameters : 1. [ result ], 2. [ source | result ], 3. [ source | result | mask ]
    "direction": args.merge_type,
    # important parameters : String of "AtoB" or "BtoA". Direction of face transformation.
}

input_fn = args.input
output_fn = args.output
duration = None
vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, opr=args.mode,duration=duration)
