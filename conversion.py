import os
import keras.backend as K
from networks.faceswap_gan_model import FaceswapGANModel
from converter.video_converter import VideoConverter
from detector.face_detector import MTCNNFaceDetector

def converter(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu_id'])
    K.set_learning_phase(0)

    # Input/Output resolution
    RESOLUTION = args['face_size']  # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

    # Architecture configuration
    arch_config = {}
    arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
    arch_config['use_self_attn'] = True
    arch_config['norm'] = "instancenorm"  # instancenorm, batchnorm, layernorm, groupnorm, none
    arch_config['model_capacity'] = "standard"  # standard, lite

    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=args['model_dir'])


    mtcnn_weights_dir = "./mtcnn_weights/"
    fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
    vc = VideoConverter()
    vc.set_face_detector(fd)
    vc.set_gan_model(model)


    options = {
        "use_smoothed_bbox": True,
        "use_kalman_filter": True,
        "use_auto_downscaling": True,
        "bbox_moving_avg_coef": 0.65,
        "min_face_area": 35 * 35,
        "IMAGE_SHAPE": model.IMAGE_SHAPE,
        "kf_noise_coef": 3e-3,
        "use_color_correction": "hist_match",
        "detec_threshold": 0.7,
        "roi_coverage": 0.9,
        "enhance": 0.,
        "output_type": args['output_type'],
        # important parameters : 1. [ result ], 2. [ source | result ], 3. [ source | result | mask ]
        "direction": args['merge_type'],
        # important parameters : String of "AtoB" or "BtoA". Direction of face transformation.
    }

    input_fn = args['input']
    output_fn = args['output']
    duration = None
    vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, opr=args['mode'],duration=duration)