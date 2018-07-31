#from .faceID_lib import faceID_lib
import os
import time
import cv2
import sys
sys.path.append(os.getcwd())
import argparse
import faceID_lib



def make_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, required=True, help="input image name")
    parser.add_argument("--output", type=str, default="examples/find_face_in_picture/result.jpg", help="output image name") 
    
    return parser



if __name__ == "__main__":
    
    parser1 = make_parser()
    args = parser1.parse_args()
    
    image = faceID_lib.load_image_file(args.input)

    start = time.time()
    sample_face_locations = faceID_lib.face_locations(image,model="cnn")    
    end = time.time()
    print (sample_face_locations)
    print ("Elapsed time: ",end-start," ms")
    
    for y1,x2,y2,x1 in sample_face_locations:
        print (x1,y1,x2,y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255,0), 2)
    
    faceID_lib.imwrite(args.output,image)
        
    faceID_lib.show_image_cv("face_detection",image,wait_time=0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    