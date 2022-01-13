import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
predict = os.listdir(args.path)
for pre in predict:
    os.rename(args.path+pre, args.path+pre[:-4] + "_pred.png")
