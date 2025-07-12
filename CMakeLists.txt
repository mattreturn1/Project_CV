cmake_minimum_required(VERSION 3.10)
project(Project_CV)

set(CMAKE_CXX_STANDARD 20)

#For Linux users
#set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

#For Windows users
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
        src/utils.cpp
        src/face_detector.cpp
        src/main.cpp
)

add_executable(Project_CV ${SOURCES})

target_link_libraries(Project_CV ${OpenCV_LIBS})