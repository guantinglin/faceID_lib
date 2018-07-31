#from .faceID_lib import faceID_lib
import os
import sys
import click
sys.path.append(os.getcwd())

import faceID_lib
import argparse




def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True, help='face sample image')
    parser.add_argument('--input', type=str, required=True, help='input folder that contains testing images')
    parser.add_argument('--output_folder', type=str, required=True, help='output folder')
    parser.add_argument('--cpus', type=int, required=True, help='number of cpus that aim to use')
    parser.add_argument('--model', type=str, required=False,default='hog', help='Face detection model, default is hog')
    
    return parser





def main(sample,input,output_folder,cpus, model):
    faceID_lib.find_ID(sample,input,output_folder,cpus, model)
    
if __name__ == '__main__':
   

    parser1 = make_parser()
    args = parser1.parse_args()

    main(args.sample,args.input,args.output_folder,args.cpus,args.model)
    
    
    
    
    
    
    
    