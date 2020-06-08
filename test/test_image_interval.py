from stsearch.interval import ImageInterval
from rekall.bounds import Bounds3D

if __name__ == "__main__":
    origin = ImageInterval.readfile("car.jpg")
    origin.savefile("test-copy.jpg")
    # make a center crop
    center_crop = ImageInterval(bounds=Bounds3D(0, 0, .25, .75, .25, .75), root=origin)
    center_crop.savefile("test-center-crop.jpg")
    # make a upper loeft crop
    UL_crop = ImageInterval(bounds=Bounds3D(0, 0, 0, .5, 0, .5), root=origin)
    UL_crop.savefile("test-UL-crop.jpg")
