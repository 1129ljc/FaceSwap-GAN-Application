import torch
import face_alignment
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
import argparse
import os

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--input-face-dir", type=str, required=True, help="Directory of extracted faces.")
args_parser.add_argument("--save-path", type=str, required=True, help="Directory of saving mask.")
args_parser.add_argument("-g","--gpu-num", type=str, required=True, help="numbers of GPU.")
args = args_parser.parse_args()


assert os.path.exists(os.path.join(args.input_face_dir,'aligned_faces'))


save_path = args.save_path
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

fns_face = glob(os.path.join(args.input_face_dir,'aligned_faces',f"*.*"))




fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:'+args.gpu_num, flip_input=False)

'''
file = '/ssd1/ljc/video_database/test/aligned_faces/frame1face0.jpg'
preds = fa.get_landmarks(file)[0]
# print(preds[0, 0], preds[0, 1])

img = cv2.imread(file)
for i in range(68):
    # print(int(preds[i, 0]), int(preds[i, 1]))
    img = cv2.circle(img, (int(preds[i, 0]), int(preds[i, 1])), 5, (0, 0, 255), -1)

cv2.imwrite('/ssd1/ljc/video_database/test/point.jpg', img)

mask = np.zeros_like(img)
pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

pnts_mouth = [(preds[i, 0], preds[i, 1]) for i in range(48, 60)]
hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

cv2.imwrite('/ssd1/ljc/video_database/test/mask1.jpg', mask)

mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)

cv2.imwrite('/ssd1/ljc/video_database/test/mask2.jpg', mask)

mask = cv2.GaussianBlur(mask, (7, 7), 0)

cv2.imwrite('/ssd1/ljc/video_database/test/mask3.jpg', mask)
'''

fns_face_not_detected = []

for j in tqdm(range(len(fns_face))):
    fn = fns_face[j]
    # print(fn)
    raw_fn = PurePath(fn).parts[-1]

    x = plt.imread(fn)
    x = cv2.resize(x, (256, 256))
    preds = fa.get_landmarks(x)

    if preds is not None:
        preds = preds[0]
        mask = np.zeros_like(x)

        # Draw right eye binary mask
        pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
        hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
        mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

        # Draw left eye binary mask
        pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
        hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
        mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

        # Draw mouth binary mask
        pnts_mouth = [(preds[i, 0], preds[i, 1]) for i in range(48, 60)]
        hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
        mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

    else:
        mask = np.zeros_like(x)
        print(f"No faces were detected in image '{fn}''")
        fns_face_not_detected.append(fn)

    plt.imsave(fname=os.path.join(save_path,raw_fn), arr=mask, format="jpg")
