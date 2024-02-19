# To configure the correct python env in CLion, add to CMake options:
# -DPython3_ROOT_DIR=<path to python env bin dir>

cmake_minimum_required(VERSION 3.2)
cmake_policy(SET CMP0057 NEW)

project(leven LANGUAGES CXX)

include_directories("${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 20)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")  # needed for pybind11
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE pybind11_DIR
)

#include(CMakePrintHelpers)
#cmake_print_variables(Python3_EXECUTABLE PYTHON_EXECUTABLE pybind11_DIR)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(leven leven.cpp)

# To compile as standalone:
# c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) leven.cpp -o leven$(python3-config --extension-suffix) -undefined dynamic_lookup
