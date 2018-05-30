YAEL_DIR = /home/yuanyong/libs/yael

CXX = c++
CXXFLAGS = -std=c++0x -O3 -g -Wall -fmessage-length=0 -fopenmp -I$(YAEL_DIR)
LDFLAGS = -L$(YAEL_DIR)/yael -lyael

all: train_PQ

train_PQ_codebook.o: train_PQ_codebook.h train_PQ_codebook.cpp 
	$(CXX) $(CXXFLAGS) -c train_PQ_codebook.cpp -o train_PQ_codebook.o

train_PQ.o: train_PQ.cpp
	$(CXX) $(CXXFLAGS) -c train_PQ.cpp -o train_PQ.o

train_PQ: train_PQ.o train_PQ_codebook.o 
	$(CXX) $(CXXFLAGS) -o train_PQ train_PQ.o train_PQ_codebook.o  $(LDFLAGS)

clean:
	rm train_PQ_codebook.o train_PQ.o train_PQ
