import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
import tqdm
from umeyama import umeyama
import mtcnn_detect_face
import argparse


args_parser = argparse.ArgumentParser()
args_parser.add_argument("--input-video", type=str, required=True, help="Directory of video for extract.")
args_parser.add_argument("--output-dir", type=str, required=True, help="Directory of output face.")
args_parser.add_argument("-g","--gpu-num", type=str, required=True, help="numbers of GPU.")
args = args_parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num


def create_mtcnn(sess, model_path):
    # variable_scope 变量作用域
    # placeholder 喂入数据的占位符
    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = mtcnn_detect_face.PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = mtcnn_detect_face.RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = mtcnn_detect_face.ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)
    return pnet, rnet, onet


WEIGHTS_PATH = "./mtcnn_weights/"

sess = K.get_session()
# 返回TensorFlow会话
with sess.as_default():
    pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

pnet = K.function([pnet.layers['data']], [pnet.layers['conv4-2'], pnet.layers['prob1']])
rnet = K.function([rnet.layers['data']], [rnet.layers['conv5-2'], rnet.layers['prob1']])
onet = K.function([onet.layers['data']], [onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])


def get_src_landmarks(x0, x1, y0, y1, pnts):
    """
    x0, x1, y0, y1: (smoothed) bbox coord.
    pnts: landmarks predicted by MTCNN
    """
    src_landmarks = [(int(pnts[i + 5] - x0), int(pnts[i] - y0)) for i in range(5)]
    return src_landmarks


def get_tar_landmarks(img):
    """
    img: detected face image
    """
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
    ]
    # 人脸五官的大概位置

    img_size = img.shape
    tar_landmarks = [(int(xy[0] * img_size[0]), int(xy[1] * img_size[1])) for xy in ratio_landmarks]
    return tar_landmarks


def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks):
    """
    umeyama(src, dst, estimate_scale)
    landmarks coord. for umeyama should be (width, height) or (y, x)
    """
    src_size = src_im.shape
    src_tmp = [(int(xy[1]), int(xy[0])) for xy in src_landmarks]
    tar_tmp = [(int(xy[1]), int(xy[0])) for xy in tar_landmarks]
    M = umeyama(np.array(src_tmp), np.array(tar_tmp), True)[0:2]
    result = cv2.warpAffine(src_im, M, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE)
    return result


def process_mtcnn_bbox(bboxes, im_shape):
    """
    output bbox coordinate of MTCNN is (y0, x0, y1, x1)
    Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
    """
    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bboxes[i, 0:4]
        w, h = int(y1 - y0), int(x1 - x0)
        length = (w + h) / 2
        center = (int((x1 + x0) / 2), int((y1 + y0) / 2))
        new_x0 = np.max([0, (center[0] - length // 2)])  # .astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0] + length // 2)])  # .astype(np.int32)
        new_y0 = np.max([0, (center[1] - length // 2)])  # .astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1] + length // 2)])  # .astype(np.int32)
        bboxes[i, 0:4] = new_x0, new_y1, new_x1, new_y0
    return bboxes

if not os.path.exists(os.path.join(args.output_dir,'aligned_faces')):
    os.makedirs(os.path.join(args.output_dir,'aligned_faces'))
if not os.path.exists(os.path.join(args.output_dir,'raw_faces')):
    os.makedirs(os.path.join(args.output_dir,'raw_faces'))


fn_input_video = args.input_video
cap = cv2.VideoCapture(fn_input_video)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm.tqdm(range(frame_num)):
    ref, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    minsize = 10  # minimum size of face
    detec_threshold = 0.7
    threshold = [0.6, 0.7, detec_threshold]  # three steps's threshold
    factor = 0.709  # scale factor
    # 大部分代码中detec_threshold，threshold，factor三个参数固定该数值

    faces, pnts = mtcnn_detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

    faces = process_mtcnn_bbox(faces, frame.shape)


    for idx, (x0, y1, x1, y0, conf_score) in enumerate(faces):
        det_face_im = frame[int(x0):int(x1), int(y0):int(y1), :]

        # get src/tar landmarks
        src_landmarks = get_src_landmarks(x0, x1, y0, y1, pnts[...,idx])
        tar_landmarks = get_tar_landmarks(det_face_im)

        # align detected face
        aligned_det_face_im = landmarks_match_mtcnn(det_face_im, src_landmarks, tar_landmarks)
        fname = os.path.join(args.output_dir,'aligned_faces',f"frame{i + 1}face{str(idx)}.jpg")
        plt.imsave(fname, aligned_det_face_im, format="jpg")
        fname = os.path.join(args.output_dir,'raw_faces',f"frame{i + 1}face{str(idx)}.jpg")
        plt.imsave(fname, det_face_im, format="jpg")

