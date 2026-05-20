
CXXFLAGS = -Wall -O3 -I ./include/ 

all: DeteccionEsquinasM DeteccionEsquinasCS

DeteccionEsquinasCS: DeteccionEsquinasCS.o
	g++ $(CXXFLAGS) -o DeteccionEsquinasCS DeteccionEsquinasCS.o `pkg-config opencv4 --libs`

DeteccionEsquinasCS.o: DeteccionEsquinasCS.cpp
	g++ $(CXXFLAGS) -c DeteccionEsquinasCS.cpp `pkg-config opencv4 --cflags`


DeteccionEsquinasM: DeteccionEsquinasM.o
	g++ $(CXXFLAGS) -o DeteccionEsquinasM DeteccionEsquinasM.o `pkg-config opencv4 --libs`

DeteccionEsquinasM.o: DeteccionEsquinasM.cpp
	g++ $(CXXFLAGS) -c DeteccionEsquinasM.cpp `pkg-config opencv4 --cflags`

clean: 
	rm DeteccionEsquinasM.o DeteccionEsquinasM DeteccionEsquinasCS.o DeteccionEsquinasCS
