cmake_minimum_required(VERSION 3.20)
project(parallel)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++2a -fopenmp")

set(CMAKE_CXX_STANDARD 17)

add_executable(parallel 30.09.cpp barrier.cpp barrier.h randomize.cpp randomize.h fib.cpp fib.h)
