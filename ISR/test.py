import numpy as np
from PIL import Image
from ISR.models import RRDN, RDN
from ISR.predict import Predictor
import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()


#model = RDN(weights='psnr-large')


weights = glob.glob(os.path.join(args.path, "rdn*.hdf5"))
print(weights)
for i in weights:

    model = RDN(arch_params={'C': 4, 'D': 12,
                             'G': 64, 'G0': 64, 'x': 3})
    predictor = Predictor(input_dir='../data/test/',
                          output_dir=f'../data/test/{i[2:]}/')
    '''predictor.get_predictions(
        model, weights_path='./rrdn-C4-D3-G64-G064-T10-x3_epoch012.hdf5')'''
    print(i)
    predictor.get_predictions(
        model, weights_path=i)
