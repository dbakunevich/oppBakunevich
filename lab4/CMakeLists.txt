cmake_minimum_required(VERSION 3.6)
project(lab4)

set(CMAKE_CXX_STANDARD 20)

include_directories($ENV{MSMPI_INC})
#find_package(MPI REQUIRED)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(lab4 main.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS})