## covdet

C++ API for covdet of VLFeat

## Dependencies

- OpenCV, 3.x or 2.x
- [VLFeat 0.9.20](http://www.vlfeat.org/), included in the project
- Armadillo, [armadillo-8.x](http://arma.sourceforge.net/download.html)

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

![](http://ose5hybez.bkt.clouddn.com/github/covdet/tower.png)
