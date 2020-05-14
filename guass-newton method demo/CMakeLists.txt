cmake_minimum_required(VERSION 3.16)
project(ch6)

set(CMAKE_CXX_STANDARD 14)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
include_directories("/usr/include/eigen3")


add_executable(ch6 main.cpp)
target_link_libraries(ch6 ${OpenCV_LIBS})
