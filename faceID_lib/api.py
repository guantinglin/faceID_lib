#import PIL.Image
import dlib
import numpy as np
import cv2
from .faceID_models import faceID_models
import itertools
import sys
import os 
import re
import multiprocessing
import shutil

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = faceID_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = faceID_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = faceID_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = faceID_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def load_image_file(file):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array using opencv
    
    :param file: image file name or file object to load    
    :return: image contents as numpy array (RGB order)
    """
        
    return cv2.cvtColor(np.array(cv2.imread(file)), cv2.COLOR_BGR2RGB)

def imwrite(file_name,image):
    """
    Write numpy array into disk
    
    :param file: image file name or file object to write    
    :return: None
    """
            
    cv2.imwrite(file_name,cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    
def face_distance(face_encodings, face_to_compare):
    """      
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
    
    
def show_image_cv(screen_name,image,wait_time=0):
    '''
    Show out image
    
    :param screen_name: Screen name of the showing window
    :param image: Numpy array of image
    :param wait_time: Delay time of the screen
    '''
    cv2.imshow(screen_name,cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait_time)




def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()    
    
def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])
    
def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
    
    
def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)
        
        
def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]        
        
def _raw_face_locations_batched(images, number_of_times_to_upsample=1, batch_size=128):
    """
    Returns an 2d array of dlib rects of human faces in a image using the cnn face detector

    :param img: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)


def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
    """
    Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector
    If you are using a GPU, this can give you much faster results since the GPU
    can process batches of images at once. If you aren't using a GPU, you don't need this function.

    :param img: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param batch_size: How many images to include in each GPU processing batch.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    def convert_cnn_detections_to_css(detections):
        return [_trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape) for face in detections]

    raw_detections_batched = _raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

    return list(map(convert_cnn_detections_to_css, raw_detections_batched))        


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]

    
        
def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="large")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    
    
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)  
        
#######################################################################################################################################
def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|JPG)', f, flags=re.I)]        

    
def test_image(target_encodings,image_to_check, model,output_folder):
    '''
    Given an unknown image, compare the face appear in it with sample encodings
    
    :param target_encodings: encodings extracted from sample image
    :param image_to_check: image path that needs to be check
    :param model: face detection model, either 'hog' or 'cnn'
    :param output_folder: output folder of images that has target_encodings in    
    
    '''
    unknown_image = load_image_file(image_to_check)
    
    unknown_image_locations = face_locations(unknown_image,model=model)
    test_encodings = face_encodings(unknown_image,unknown_image_locations)
    
    for idx,test_encoding in enumerate(test_encodings):
        compare_results = compare_faces(target_encodings,test_encoding,0.6)        
        print (compare_results)
        if any(compare_results):
            shutil.copyfile(image_to_check, os.path.join(output_folder,os.path.basename(image_to_check)))




        
def process_images_in_process_pool(target_encodings,images_to_check, number_of_cpus, model,output_folder):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        itertools.repeat(target_encodings),
        images_to_check,
        itertools.repeat(model),
        itertools.repeat(output_folder)
    )

    pool.starmap(test_image, function_parameters)        
                
        
        
def find_ID(sample,input,output_folder,cpus, model):
    '''
    find the images that has at lease one ID that appear in sample image
    
    :param sample: sample image path
    :param input: folder contains testing images
    :param cpus: number of cpus that you want to use
    :param model: face detection model, either 'hog' or 'cnn'
    
    '''
    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1
        
    if not (os.path.exists(input) or os.path.exists(sample)):
        click.echo("ERROR: " + input + " is not found")
        sys.exit(1)        
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    try:
        image_sample = load_image_file(sample)
        target_locations = face_locations(image_sample,model=model)
        target_encodings = face_encodings(image_sample,target_locations)
        print (target_encodings)
        print(len(target_encodings)," faces are found in sample image")
            
    except:
        print("Cannot encode from the sample image")
        quit()
    
    if os.path.isdir(input):
        if cpus == 1:
            [test_image(target_encodings,image_file, model,output_folder) for image_file in image_files_in_folder(input)]
        else:
            process_images_in_process_pool(target_encodings,image_files_in_folder(input), cpus, model,output_folder)    
    else:
        test_image(target_encodings,image_to_check, model,output_folder)        

        
    
    
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        