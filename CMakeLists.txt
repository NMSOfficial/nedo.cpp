cmake_minimum_required(VERSION 3.20)
project(nedo_cpp VERSION 0.1.0 LANGUAGES CXX)

option(NEDO_BUILD_PYTHON "Build Python extension" ON)
option(NEDO_BUILD_TESTS "Build tests" ON)
option(NEDO_NATIVE "Enable -march=native where supported" ON)
option(NEDO_OPENMP "Enable OpenMP" ON)
option(NEDO_FAST_MATH "Enable aggressive floating-point reassociation (may affect parity)" OFF)

# Portable wheels set NEDO_NATIVE=OFF. Local users can request a host-tuned
# source build without editing pyproject.toml.
if(DEFINED ENV{NEDO_NATIVE_BUILD} AND NOT "$ENV{NEDO_NATIVE_BUILD}" STREQUAL "0")
  set(NEDO_NATIVE ON CACHE BOOL "Enable host-native ISA" FORCE)
endif()
if(DEFINED ENV{NEDO_FAST_MATH_BUILD} AND NOT "$ENV{NEDO_FAST_MATH_BUILD}" STREQUAL "0")
  set(NEDO_FAST_MATH ON CACHE BOOL "Enable fast math" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nedo_core
  src/gguf.cpp
  src/kernels.cpp
  src/tokenizer.cpp
  src/model.cpp
)
target_include_directories(nedo_core PUBLIC include)

if(MSVC)
  target_compile_options(nedo_core PRIVATE /O2 /Ob3 /DNOMINMAX)
  if(NEDO_NATIVE AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    target_compile_options(nedo_core PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(nedo_core PRIVATE -O3 -fno-math-errno -fno-trapping-math)
  if(NEDO_FAST_MATH)
    target_compile_options(nedo_core PRIVATE -ffast-math)
  endif()
  if(NEDO_NATIVE)
    target_compile_options(nedo_core PRIVATE -march=native)
  endif()
endif()

if(NEDO_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nedo_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

add_executable(nedo-inspect src/inspect.cpp)
target_link_libraries(nedo-inspect PRIVATE nedo_core)

if(NEDO_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core src/pybind.cpp)
  target_link_libraries(_core PRIVATE nedo_core)
  install(TARGETS _core LIBRARY DESTINATION nedo RUNTIME DESTINATION nedo)
endif()

if(NEDO_BUILD_TESTS)
  enable_testing()
  add_executable(nedo-kernel-test tests/test_kernels.cpp)
  target_link_libraries(nedo-kernel-test PRIVATE nedo_core)
  add_test(NAME kernels COMMAND nedo-kernel-test)
endif()
