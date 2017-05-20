cmake_minimum_required(VERSION 3.5)
project(q_learning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

set(SOURCE_FILES main.c)
add_executable(q_learning ${SOURCE_FILES})