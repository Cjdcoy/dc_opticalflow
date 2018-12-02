opticalflow_utilities
======================
The goal of opticalflow_utilies is to provide a pool of scripts that allows you to do whaterver you want with many different 
kinds of optical flows. For now the optical flow algorithms provided are from <a href="https://github.com/lmb-freiburg/flownet2" target="_blank">flownet2</a><br>.


### Install (need apt-get)

this installation script has been made on ubuntu 16.04

```
sudo apt-get update
bash build.bash
```

End with an example of getting some data out of the system or using it for a little demo


## Features

### Local:

- [x] .png to .flo
- [x] .flo to .png in the same program (input IMG IMG)
- [x] video to video of opticalFlow
- [ ] videos to videos of opticalFlow
- [x] opticalflow in realtime
- [ ] save the reatime flow into a file

### Cloud:
- [x] send webcam stream, server compute opticalflow and return the result to the client
- [x] send x videos, server compute opticalflow and return the result to the client
- [x] send x images, server compute opticalflow and return the result to the client
- [x] client is able to save anything the server returns

### Global
- [ ] add the choice to select any algorithm

![](https://github.com/Cjdcoy/opticalflow_utilities/blob/master/documents/OF.gif)

