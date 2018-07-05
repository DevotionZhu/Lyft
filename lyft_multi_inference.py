import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

##--------------------------
# my libraries from env  OD-lab-py35
##--------------------------
import time
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

#import helper_cityscapes as helper
#import helper_lyft as helper
import numpy as np

import warnings
import scipy.misc
import tensorflow as tf


from datetime import timedelta
from distutils.version import LooseVersion

import cv2


file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

#--------------------------
# Define global variables 
#--------------------------

cropped_zeros = np.zeros((100, 800))

IMAGE_SHAPE = (256, 512)


#--------------------------
# Define encoder function
#--------------------------


def cv2_encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()

    #if pil_img.mode != 'RGB':
    #    pil_img = pil_img.convert('RGB')

    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def single_img_inference(img, image_shape, sess, image_place_holder, logits, keep_prob):

    #print("input image shape:", img.shape)        
    # (600, 800)
    #showimg(img)
    
    
    #street_im = scipy.misc.toimage(img)
        
        
    # (400, 800)
    crop_img = img[100:500,:,:]
    #showimg(crop_img)
    
    # (400, 800) to (256, 512)
    image = scipy.misc.imresize(crop_img, image_shape)

    
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input: [image]})
    
    ###################
    
    # Road
    im_softmax_r = im_softmax[0][:, 0].reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1]) # (256, 512)
    
    #print("im_softmax_r shape", im_softmax_r.shape) #im_softmax_r shape (256, 512)
    #print( "road max:", np.max(im_softmax_r))
        
    # reshape: (131072,) into shape (256, 512)  != resize: (256, 512) to (400, 800)
    #im_softmax_r = im_softmax_r.reshape(crop_img.shape[0], crop_img.shape[1])
    im_softmax_r_1 =  (im_softmax_r > 0.5).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1) # (256, 512, 1)
        
    # binary 0, 1:  im_softmax_r_1.astype(np.uint8)
        
    #print("road_img_int = :", road_img_int) # each element is either 0=black or 255:white
    road_img_int = im_softmax_r_1.astype(np.uint8)*255
    #showim(road_img_int)
        
    img_prep = np.concatenate((road_img_int, road_img_int, road_img_int), axis =2)
    resize_road = scipy.misc.imresize( img_prep , crop_img.shape) # (  400, 800 , 3)
    #showimg(resize_road)  # (400, 800, 3)

    # 0 and 255
    resize_road_int = resize_road[:,:,0]  # (400, 800)
    padding_road_int = np.concatenate((cropped_zeros, resize_road_int),axis=0)
    padding_road_int = np.concatenate((padding_road_int,cropped_zeros),axis=0) # (600,800)

    #padding_road_int_1 = padding_road_int[:, :, np.newaxis]
    #showim(padding_road_int_1)
        
    #mask = np.dot(padding_road_int_1, np.array([[128, 64, 128, 64]])) 
    #mask = scipy.misc.toimage(mask, mode="RGBA")
    #showimg(mask)
    #street_im.paste(mask, box=None, mask=mask)
    #showimg(street_im)
    
    # binary 0, 1:  
    padding_road_int[padding_road_int == 255]=1
    binary_padding_road = padding_road_int.astype('uint8')
    #print("binary_padding_road: ", binary_padding_road.shape, binary_padding_road)
    

    ###########################

    
    # 1 fo car
    im_softmax_r = im_softmax[0][:, 1].reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    im_softmax_r_1 =  (im_softmax_r > 0.5).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1) # (256, 512, 1)
    car_img_int = im_softmax_r_1.astype(np.uint8)*255
    #showim(car_img_int)
        
    img_prep = np.concatenate((car_img_int, car_img_int, car_img_int), axis =2)
    resize_car = scipy.misc.imresize( img_prep , crop_img.shape) # (  400, 800 , 3)
    #showimg(resize_car)  # (400, 800, 3)
        
    # 0 and 255
    resize_car_int = resize_car[:,:,0]  # (400, 800)
    padding_car_int = np.concatenate((cropped_zeros, resize_car_int),axis=0)
    padding_car_int = np.concatenate((padding_car_int,cropped_zeros),axis=0) # (600,800)
    
    #padding_car_int_1 = padding_car_int[:, :, np.newaxis]
    #showim(padding_car_int_1)
        
    # binary 0, 1:  
    #binary_padding_car
    padding_car_int[padding_car_int == 255]=1
    binary_padding_car = padding_car_int.astype('uint8')
    #print("binary_padding_car: ", binary_padding_car.shape, binary_padding_car)
    

        
    #mask = np.dot(padding_car_int_1, np.array([[0, 0, 142, 64]]))
    #mask = scipy.misc.toimage(mask, mode="RGBA")
    #showimg(mask)
    #street_im.paste(mask, box=None, mask=mask)
    #showimg(street_im)  


    ##################################
    
    #scipy.misc.imsave(os.path.join(output_dir), street_im)
    
    # 0, 1
    return binary_padding_road, binary_padding_car #(600, 800)


#--------------------------
#  restore saved checkpoint, model
#--------------------------


#--------------------------
#  mainstream pipeline
#--------------------------

video = skvideo.io.vread(file)


answer_key = {}



new_saver = tf.train.import_meta_graph('./models_lyft/20epoch/cont_epoch_19.ckpt.meta')
#new_saver = tf.train.import_meta_graph('./models_lyft/25epoch/cont_epoch_24.ckpt.meta')

graph = tf.get_default_graph()

infer_batch_size = 16 #64 #128 #32 #16 #4 #1 #4 #16

#binary_road_single = []
#binary_car_single = []
    
    
binary_road_test1 = []
binary_car_test1 = []

binary_road_batch = []
binary_car_batch = []

binary_road_single = []
binary_car_single = []
    
with tf.Session() as sess:
    new_saver.restore(sess, "./models_lyft/20epoch/cont_epoch_19.ckpt")
    #new_saver.restore(sess, "./models_lyft/25epoch/cont_epoch_24.ckpt")
    
    input = graph.get_tensor_by_name('image_input:0')
    #print(input)
    logits = sess.graph.get_tensor_by_name('logits:0')
    #print(logits)
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    #print(keep_prob)
    image_place_holder = input

    X_arr = np.zeros((infer_batch_size, 256, 512, 3), dtype= np.uint8)
        
    for batch_i in range(0, len(video), infer_batch_size):
        cnt = 0 
        for j in range(batch_i, min(batch_i + infer_batch_size, len(video))):
            
            #road_binary, car_binary = single_img_inference(video[j], IMAGE_SHAPE, sess, image_place_holder, logits, keep_prob)

            #binary_road_test1.append(road_binary)
            #binary_car_test1.append(car_binary)
            
            X_arr[cnt, :, :, :] = scipy.misc.imresize(video[j][100:500, :, :], (256, 512, 3))
            cnt += 1
            
            
        road_test1 = np.zeros((cnt, 256, 512, 1), dtype= np.float32)
        car_test1 = np.zeros((cnt, 256, 512, 1), dtype= np.float32)
        
        im_softmax_list = sess.run([tf.nn.softmax(logits)],  feed_dict = { keep_prob: 1.0, input:X_arr[0:cnt] })
        
        # 4? x 256, 512, 3
        im_softmax_array = im_softmax_list[0].reshape(cnt, IMAGE_SHAPE[0], IMAGE_SHAPE[1], -1)

        for i in range(cnt):
            # road 
            im_softmax = im_softmax_array[i] # im_softmax_array shape  256 x 512 x 3 !!
            
            road_test1[i] = im_softmax[:,:,0].reshape(*im_softmax[:,:,0].shape, 1) #(256 x 512 x 1)
            car_test1[i] = im_softmax[:,:,1].reshape(*im_softmax[:,:,1].shape, 1) #(256 x 512 x 1)
            
            road_img_int = (road_test1[i] > 0.5).astype(np.uint8)*255 #  256, 512, 1
            car_img_int = (car_test1[i] > 0.5).astype(np.uint8)*255 #  256, 512, 1

            img_prep1 = np.concatenate((road_img_int, road_img_int, road_img_int), axis =2)
            img_prep2 = np.concatenate((car_img_int, car_img_int, car_img_int), axis =2)
            
            resize_road = scipy.misc.imresize( img_prep1 , (400,800,3)) # (  400, 800 , 3)
            resize_car = scipy.misc.imresize( img_prep2 , (400,800,3)) # (  400, 800 , 3)
            
            resize_road_int = resize_road[:,:,0]  # (400, 800)
            padding_road_int = np.concatenate((cropped_zeros, resize_road_int),axis=0)
            padding_road_int = np.concatenate((padding_road_int,cropped_zeros),axis=0) # (600,800)
            resize_car_int = resize_car[:,:,0]  # (400, 800)
            padding_car_int = np.concatenate((cropped_zeros, resize_car_int),axis=0)
            padding_car_int = np.concatenate((padding_car_int,cropped_zeros),axis=0) # (600,800)
            # binary 0, 1:  
            padding_road_int[padding_road_int == 255]=1
            road_binary_test1 = padding_road_int.astype('uint8')

            binary_road_test1.append(road_binary_test1)
           # binary 0, 1:  
            padding_car_int[padding_car_int == 255]=1
            #binary_padding_car.append(padding_car_int.astype('uint8'))
            car_binary_test1 = padding_car_int.astype('uint8')
            binary_car_test1.append(car_binary_test1)

    # Frame numbering starts at 1
    frame = 1
    for k in range(0, len(binary_road_test1)):
        answer_key[frame] = [cv2_encode(binary_car_test1[k]), cv2_encode(binary_road_test1[k])]
        frame+=1

 
    
    
#with open('batch_1.log', 'w') as outfile:
#    json.dump(json.dumps(answer_key), outfile)
# Print output in proper json format
print (json.dumps(answer_key)) 