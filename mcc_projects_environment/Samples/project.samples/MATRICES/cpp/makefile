CXX=clang++
CXXFLAGS=-c -Wall -g -pedantic
LFLAGS=
OBJS=main.o mx.o

all: main.out

main.out: $(OBJS)
	$(CXX) $(LFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	clear; rm -f *.o a.out vgcore*