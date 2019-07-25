cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tts_native_inference)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(ESPEAKNG REQUIRED espeak-ng)

link_directories(native_inference ${ESPEAKNG_LIBRARY_DIRS})

add_executable(native_inference main.cpp)
include_directories(native_inference ${ESPEAKNG_INCLUDE_DIRS})
target_link_libraries(native_inference ${TORCH_LIBRARIES} ${ESPEAKNG_LIBRARIES})
set_property(TARGET native_inference PROPERTY CXX_STANDARD 14)
