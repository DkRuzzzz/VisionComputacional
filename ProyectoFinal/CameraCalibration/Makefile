
CXXFLAGS =  -Wall -g -I ../include/

all: correctImages camera_calibration

camera_calibration: objs/camera_calibration.o
	g++ $(CXXFLAGS) -o camera_calibration objs/camera_calibration.o `pkg-config opencv4 --libs`

objs/camera_calibration.o: camera_calibration.cpp
	g++ $(CXXFLAGS) -o objs/camera_calibration.o -c camera_calibration.cpp `pkg-config opencv4 --cflags`

correctImages: objs/correctImages.o
	g++ $(CXXFLAGS) -o correctImages objs/correctImages.o `pkg-config opencv4 --libs`

objs/correctImages.o: correctImages.cpp
	g++ $(CXXFLAGS) -o objs/correctImages.o -c correctImages.cpp `pkg-config opencv4 --cflags`

clean: 
	rm objs/*.o camera_calibration correctImages
