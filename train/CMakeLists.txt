cmake_minimum_required(VERSION 3.10)
project(ODEDSN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# debug and release configs
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# torch
set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(pybind11 REQUIRED)

# source files
file(GLOB SOURCE_FILES ./src/*.cpp)

# include files
include_directories(include json/include rapidcsv/src netmodels/include)

add_executable(train-astress-opt train-astress.cpp ${SOURCE_FILES})
add_executable(test-astress-opt test-astress.cpp ${SOURCE_FILES})
add_executable(test-loss test-loss.cpp ${SOURCE_FILES})

# link torch
target_link_libraries(train-astress-opt ${TORCH_LIBRARIES})
target_link_libraries(test-astress-opt ${TORCH_LIBRARIES})
target_link_libraries(test-loss ${TORCH_LIBRARIES})

target_compile_definitions(train-astress-opt PRIVATE $<$<CONFIG:Debug>:DEBUG_MODE>)
target_compile_definitions(test-astress-opt PRIVATE $<$<CONFIG:Debug>:DEBUG_MODE>)
target_compile_definitions(test-loss PRIVATE $<$<CONFIG:Debug>:DEBUG_MODE>)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()


# --------- Python Module Part -----------
pybind11_add_module(optim_module optim_module.cpp ${SOURCE_FILES})
target_link_libraries(optim_module PRIVATE ${TORCH_LIBRARIES})