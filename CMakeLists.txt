cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ChessAI)

list(APPEND CMAKE_PREFIX_PATH "~/Library/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories - add PyTorch includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})  

# Add source files
set(SOURCES
    chess_position_encoding.cpp
    chess_neural_network.cpp
    main.cpp
)

# Create executable
add_executable(chess_ai ${SOURCES})

# Link against LibTorch
target_link_libraries(chess_ai ${TORCH_LIBRARIES})

set_property(TARGET chess_ai PROPERTY CXX_STANDARD 17)

# Set compiler flags
if(MSVC)
  target_compile_options(chess_ai PRIVATE /W4)
else()
  target_compile_options(chess_ai PRIVATE -Wall -Wextra)
endif()