opticalflow_utilities
======================
The goal of opticalflow_utilies is to provide a pool of scripts that allows you to do whaterver you want with many different 
kinds of optical flows. For now the optical flow algorithms provided are from <a href="https://github.com/lmb-freiburg/flownet2" target="_blank">flownet2</a><br>.


## Easy Install FlowNet2 (need apt-get)

this installation script has been made on ubuntu 16.04.
Its use is to easily install Flownet 2 caffe models as well as python opencv, numpy....

```
sudo apt-get update
bash build.bash
```

## Usage

Utilities are in the 'solutions' folder at root.<br /> 
The 'modules' folder at root contain different types of optic flow algorithms that you can easily select.

### 1 - add/select a module
add a file called ```vision_module.py``` next to a solution
#### Examples:
```
cp modules/vision_module_flownet.py solutions/realtime/vision_module.py
cp modules/vision_module_flownet.py solutions/videos_to_computed_videos/vision_module.py
```

### 2 - run solutions
Each solution has many configuration parameters that you can see using ```-h``` flag
#### Examples:
Apply optic flow in realtime from webcam flux, previsualize and save the flux.<br /> 
```
cd solutions/realtime
python2 realtime_vision.py -h
python2 realtime_vision.py -pre 2 -s save.avi
```
Apply optic flow on a video dataset, previsualize, save each video in save_here folder flux + estimate compute time.<br /> 
```
cd solutions/videos_to_computed_videos
cp ../../modules/vision_module_absdiff.py vision_module.py
python3 videos_vision.py -h
python3 videos_vision.py -pre 1 -e 2 -l video_list_example -s save_here
```


## Features

### Local
- [x] .png to .flo
- [x] .flo to .png in the same program (input IMG IMG)
- [x] video to video of opticalFlow
- [x] videos to videos of opticalFlow
- [x] opticalflow in realtime
- [x] save the reatime flow into a file

### Cloud
- [x] send webcam stream, server compute opticalflow and return the result to the client
- [x] send x videos, server compute opticalflow and return the result to the client
- [x] send x images, server compute opticalflow and return the result to the client
- [x] client is able to save anything the server returns

### Global
- [x] add the choice to select any algorithm


![](https://github.com/Cjdcoy/opticalflow_utilities/blob/master/documents/OF.gif)

