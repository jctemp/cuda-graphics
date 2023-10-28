cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(cuda-graphics LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA) # check if cuda is present

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
file(CREATE_LINK 
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json 
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    SYMBOLIC)

#####################################################################

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

message("===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("  ${PROJECT_NAME} version : ${PROJECT_VERSION}")
message("===================================================")
message("  CMake version           : ${CMAKE_VERSION}")
message("")
message("  CUDA compiler ID        : ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message("  CUDA compiler           : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec      : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags      : ${CMAKE_CUDA_FLAGS}")
message("")
message("  C++ Compiler ID         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message("  C++ Compiler            : ${CMAKE_CXX_COMPILER}")
message("")
message("  CUDAToolkit version     : ${CUDAToolkit_VERSION}")
message("  GLEW version            : ${GLEW_VERSION}")
message("  OpenGL found            : ${OPENGL_FOUND}")
message("  GLUT found              : ${GLUT_FOUND}")
message("===================================================")

#####################################################################

add_executable(${PROJECT_NAME} 
    main.cpp
    kernel.cu
    kernel.h
    graphics.cpp
    graphics.h)
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES OFF)
target_link_libraries(${PROJECT_NAME} PRIVATE
    -static-libgcc -static-libstdc++
    CUDA::cudart_static
    CUDA::cuda_driver

    OpenGL::OpenGL
    GLEW::GLEW
    GLUT::GLUT)

