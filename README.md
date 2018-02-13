## Covdet

C++ API for covdet of VLFeat

## Algorithm

- geo_verification.hpp, RANSAC based method, see demo.cpp
- SVF.cpp, Hough voting based method, the demo can be found in image_match.cpp

## Dependencies

- OpenCV, 3.x or 2.x.
- [VLFeat 0.9.20](http://www.vlfeat.org/), has already included in the project
- Armadillo, [armadillo-8.x](http://arma.sourceforge.net/download.html).

You must install OpenCV and Armadillo, then you can compile it successfully.

## Build

```sh
cd covdet
mkdir build && cd build
cmake ..
make
```

After finished compiling, just run:

```sh
./main
```

## Matching Result

The following images show the matching results after geometry verificaiton:

![](http://ose5hybez.bkt.clouddn.com/github/covdet/brand.png)

![](http://ose5hybez.bkt.clouddn.com/github/covdet/wine.png)
