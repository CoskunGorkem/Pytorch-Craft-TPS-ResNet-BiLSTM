import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from recognition_model import Model
from utils import CTCLabelConverter, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--model_dir', default='weights/TPS-ResNet-BiLSTM-Attn.pth', type=str, help='pretrained refiner model')

global args
args = parser.parse_args()


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def recognition(image):

    converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)
    if args.rgb:
        args.input_channel = 3
    model = Model(args)

    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.model_dir, map_location=device))
    
    transformer = resizeNormalize((100, 32))
    
    #Convert RGB images to Gray Scale which is neccesary for our convolution layers.
    if image.mode == 'RGB':
        image = image.convert('L')
        
    image = transformer(image)
    batch_size = image.size(0)
    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()


    length_for_pred = torch.IntTensor([args.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, args.batch_max_length + 1).fill_(0).to(device)
    preds = model(image, text_for_pred, is_train=False)

    _, preds_index = preds.max(2)

    preds_str = converter.decode(preds_index, length_for_pred)
    preds_prob = F.softmax(preds, dim=2)

    text_prediction = preds_str[0].replace("[s]", "")
    
    return text_prediction
    
 
