# Makefile for Kalman Filter Milestone-2
CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

all: lkf ekf

lkf: src/lkf.cpp
	$(CXX) $(CXXFLAGS) -o lkf src/lkf.cpp

ekf: src/ekf.cpp
	$(CXX) $(CXXFLAGS) -o ekf src/ekf.cpp

run_lkf: lkf
	./lkf

run_ekf: ekf
	./ekf

clean:
	rm -f lkf ekf
