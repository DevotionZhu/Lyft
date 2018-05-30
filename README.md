# Lyft
Udacity-Lyft road-car semantic segmentation for Carla simulator

to compete with other udacity students for Udacity-Lyft Road-Car Carla simulator semantic segmentation !

### Setup
##### github
https://github.com/byronrwth/Lyft.git

##### Frameworks and Packages

my Anaconda environment requirements:
requirements_OD-lab-py35.txt
 
##### Dataset
multiple dataset for training from Carla simulator

##### JWT token
need for a token to submit

### Start

##### Implement
Implement the code in the /home/workspace/Example/lyft_multi.py

##### Run
Run the following command to run the project:
```
cd /home/workspace
python Example/lyft_multi.py /Example/test_video.mp4


```


### Submission
tester 'Example/lyft_multi.py'
grader 'Example/lyft_multi.py'
submit
 
 ### Tips
- The link for the frozen `VGG16` model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
-- use batch inference to increase infere speed from 3.8 FPS to 9FPS, with inference batch size = 16

-- crop origin image (600, 800) to (400, 800) and resize to (256,512) for VGG input size, remember decrop back to (600, 800) before genereate final raod binary array and car binary array for score
