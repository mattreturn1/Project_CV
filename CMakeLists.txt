cmake_minimum_required(VERSION 3.10)
project(Project_CV)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
        src/yolo_converter.cpp
        src/utils.cpp
        src/face_detector.cpp
        src/main.cpp
        src/evaluation.cpp
)

add_executable(Project_CV ${SOURCES})

target_link_libraries(Project_CV ${OpenCV_LIBS})