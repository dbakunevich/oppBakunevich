cmake_minimum_required(VERSION 3.6)
project(lab2)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")

include_directories($ENV{MSMPI_INC})

SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(lab1 main.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS})