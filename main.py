'''
CRAFT Text Detection and Text Recognition
'''

import os
import sys
import cv2
import time
import json
import torch
import imgproc
import zipfile
import argparse
import file_utils
import pytesseract
import craft_utils
import numpy as np
import torch.nn as nn
import text_recognition
import torch.backends.cudnn as cudnn

import traceback


from PIL import Image
from skimage import io
from craft import CRAFT
from torch.autograd import Variable
from collections import OrderedDict

global args

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
    

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text
    
    
    
def analysis(image_path, result_path):
    """ For test images in a folder """
    net = CRAFT()     # initialize

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net,
                                         image,
                                         args.text_threshold,
                                         args.link_threshold,
                                         args.low_text,
                                         args.cuda,
                                         args.poly,
                                         refine_net)
                                         
    opencv_image = cv2.imread(image_path)
    
    for index, box in enumerate(polys):
        xmin, xmax, ymin, ymax = box[0][0], box[1][0], box[0][1], box[2][1]
        multiplier_area = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        try:
            im_pil = Image.fromarray(multiplier_area)
            #if you want to detect the text on the image
            if args.ocr_on:
                if args.ocr_method == 'pytesseract':
                    configuration = ("-l eng --oem 1 --psm 7")
                    multiplier = (pytesseract.image_to_string(multiplier_area, config=configuration).lower())
                    multiplier = multiplier.split("\n")[0]
                    
                elif args.ocr_method == 'TPS-ResNet-BiLSTM':
                    multiplier = text_recognition.recognition(im_pil)
                    
                cv2.putText(opencv_image, multiplier, (int(xmin),int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                
            cv2.rectangle(opencv_image,(int(xmin),int(ymin)), (int(xmax),int(ymax)),(0,0,255), 1)
            cv2.imwrite(result_path, opencv_image)
                
        except:
            print("====ERROR====", traceback.format_exc())
            pass


            
            

def main(input_path):
    dashed_line = '=' * 95
    title = f'{"OCR METHOD"}:{args.ocr_method:20s}\t{"OCR ON"}:{args.ocr_on}\t{"TEXT DETECTION MODEL"}:{args.trained_model}'
    print(f'{dashed_line}\n{title}\n{dashed_line}')
    

    result_folder = './results/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    for image_name in os.listdir(input_path):
        if "jpg" in image_name or "png" in image_name or "jpeg" in image_name:
            image_path = os.path.join(input_path, image_name)
            image_name = image_path.split("/")
            image_name = image_name[-1]
            print("Image:", image_name)
            result_path = os.path.join(result_folder, image_name)
            
            analysis(image_path, result_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection and Text Recognition')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='test_images/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--input_path', default='test_images', type=str, help='test image path')
    parser.add_argument('--ocr_on', default=True, type=str, help='Turn on/off the ocr module')
    parser.add_argument('--ocr_method', default='pytesseract', type=str, help='ocr_method "pytesseract|TPS-ResNet-BiLSTM')
    
    args = parser.parse_args()
    
    main(args.input_path)
