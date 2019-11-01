# YOLO-object-detection
YOLO (You Only Look Once) introduced in 2015 and it is a state-of-the-art, real-time object detection algorithm. In the following, we will apply the YOLO algorithm to detect objects in images, videos and webcam ([Paper](https://arxiv.org/abs/1506.02640)).

## How it works?

All prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

![yolo1](https://user-images.githubusercontent.com/31302531/67925204-f9d41400-fbc7-11e9-9999-258eaed2479b.png)

This model has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. 
This makes it extremely fast, more than 1000x faster than [R-CNN](https://github.com/rbgirshick/rcnn) and 100x faster than Fast [R-CNN](https://github.com/rbgirshick/fast-rcnn). 
For detection YOLO use Darknet which prints out the objects it detected, its confidence, and how long it took to find them. 
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. For more information see the [Darknet project website](https://pjreddie.com/darknet/).

YOLO has gone through a number of different iterations, including [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (i.e., YOLOv2), capable of detecting over 9,000 object detectors [full list of classes](https://github.com/pjreddie/darknet/blob/master/data/9k.names).

**note:** YOLO9000 has low mAP and the accuracy is not quite what we would desire.

In 2018, a new version of YOLO, [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) published. YOLOv3 is significantly larger than previous models but is, in my opinion, the best one yet out of the YOLO family of object detectors.

by default We’ll be using YOLOv3 However, you can download and use another versions and models of this.

YOLO was trained on the [COCO dataset](cocodataset.org) which consists of 80 labels, including, but not limited to:

- People
- Bicycles
- Cars and trucks
- Airplanes
- Stop signs and fire hydrants
- Animals, including cats, dogs, birds, horses, cows, and sheep, to name a few
- Kitchen and dining objects, such as wine glasses, cups, forks, knives, spoons, etc. ([Full list of classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names))

## Dependencies

In order to use YOLO, some dependencies are needed:
- Python3
- OpenCV >=3.4
- Darknet 
- some python libraries e.g numpy

## Detection using a pre-trained model (with OpenCV)

By default, Darknet uses ```stb_image.h``` for image loading. If you want more support for weird formats you can use OpenCV instead. OpenCV also allows you to view images and detections without having to save them to disk.

If you don't already have installed python and OpenCV on your computer, you should do that first using [this](https://github.com/amirmohammadii/OpenCV-Installation) tutorial. 

Now do the following steps:

1. Open ```cmd``` and run ```git clone https://github.com/amirmohammadii/YOLO-object_detection```.

2. For using a pre-trained model, you need to download ```yolov3.cfg``` and ```yolov3.weights```.
   Download ```yolov3.weights``` and it's proportionate ```.cfg``` file. Then put it in ```YOLO Detection/yolo-coco``` folder.
   Also, you can use the following commands:
   
   ```wget https://pjreddie.com/media/files/yolov3.weight
   
      wget https://pjreddie.com/media/files/yolov3.cfg
   ```
   
   **note:** In order to use other versions of YOLO, you can download proportionate ```.cfg``` and ```.weights``` files from [this](https://pjreddie.com/darknet/yolo/#demo) website.
   
3. Go through ``` YOLO Detection``` directory in ```cmd```. 

4. Run ```python image_detection.py --image images/<YOUR_IMAGE> --yolo yolo-coco```.

## Detection using a pre-trained model (without OpenCV)

first you need to install Darknet. for this purpose use [this](https://pjreddie.com/darknet/install/) guide. Also, you can run the following instructions:

```
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```

If this works you should see a whole bunch of compiling information fly by:

```
mkdir -p obj
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
.....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast -lm....
```
If everything seems to have compiled correctly, try running it by ```./darknet```. As a result, You should get ```usage: ./darknet <function> ```as the output.


Now do the following steps:

1. You will have to download the pre-trained weight file (if you didn't do it before).

   ```wget https://pjreddie.com/media/files/yolov3.weights```
   
2. Then run the detector:

   ```./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg```
   
   You will see some output like this:
   
   ```
   layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    .......
    105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
    106 detection
    truth_thresh: Using default '1.000000'
    Loading weights from yolov3.weights...Done!
    data/dog.jpg: Predicted in 0.029329 seconds.
    dog: 99%
    truck: 93%
    bicycle: 99%
   ```
   
   **note:** Darknet prints out the objects it detected, its confidence, and how long it took to find them. We didn't compile Darknet with OpenCV so it can't display the detections directly. Instead, it saves them in predictions.png. You can open it to see the detected objects. Since we are using Darknet on the CPU it takes around 6-12 seconds per image. If we use the GPU version it would be much faster.


### Compiling With CUDA

Darknet on the CPU is fast but it's like 500 times faster on GPU! You'll have to have an Nvidia GPU and you'll have to install CUDA. I won't go into CUDA installation in detail because it is terrifying.

Once you have CUDA installed, change the first line of the Makefile in the base directory to read ```GPU=1```.
If you compiled using CUDA but want to do CPU computation for whatever reason you can use -nogpu to use the CPU instead:
```./darknet -nogpu imagenet test cfg/alexnet.cfg alexnet.weights```

## Detection in video streams 

Now that we’ve learned how to apply the YOLO object detector to single images, let’s also utilize YOLO to perform object detection in input video files as well.

1. Open ```cmd``` and go through ``` YOLO Detection``` directory.

2. Run ```python  yolo_video.py  --input  videos/car.mp4   --output  output/car.avi  --yolo  yolo-coco```

## Real-Time Detection on a Webcam

Running YOLO on test data isn't very interesting if you can't see the result. Instead of running it on a bunch of images let's run it on the input from a webcam!

To run this demo you will need to compile Darknet with CUDA and OpenCV. Then run the command:

```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```

You will need a webcam connected to the computer that OpenCV can connect to or it won't work. If you have multiple webcams connected and want to select which one to use you can pass the flag ```-c <num>``` to pick (OpenCV uses webcam ```0``` by default).
