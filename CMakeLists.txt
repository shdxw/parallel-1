cmake_minimum_required(VERSION 3.0)
project(parallel)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++2a -fopenmp")

set(CMAKE_CXX_STANDARD 17)

add_executable(parallel main.cpp barrier.cpp barrier.h randomize.cpp randomize.h fib.cpp fib.h)
