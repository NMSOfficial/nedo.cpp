cmake_minimum_required(VERSION 3.20)
project(nedo_cpp VERSION 0.1.0 LANGUAGES CXX)

option(NEDO_BUILD_PYTHON "Build Python extension" ON)
option(NEDO_BUILD_TESTS "Build tests" ON)
option(NEDO_NATIVE "Enable -march=native where supported" ON)
option(NEDO_OPENMP "Enable OpenMP" ON)
option(NEDO_FAST_MATH "Enable aggressive floating-point reassociation (may affect parity)" OFF)
option(NEDO_CUDA "Enable CUDA backend when a CUDA compiler is available" ON)

# Portable wheels set NEDO_NATIVE=OFF. Local users can request a host-tuned
# source build without editing pyproject.toml.
if(DEFINED ENV{NEDO_NATIVE_BUILD} AND NOT "$ENV{NEDO_NATIVE_BUILD}" STREQUAL "0")
  set(NEDO_NATIVE ON CACHE BOOL "Enable host-native ISA" FORCE)
endif()
if(DEFINED ENV{NEDO_FAST_MATH_BUILD} AND NOT "$ENV{NEDO_FAST_MATH_BUILD}" STREQUAL "0")
  set(NEDO_FAST_MATH ON CACHE BOOL "Enable fast math" FORCE)
endif()

set(NEDO_CUDA_REQUIRED OFF)
if(DEFINED ENV{NEDO_CUDA_BUILD})
  if("$ENV{NEDO_CUDA_BUILD}" STREQUAL "0")
    set(NEDO_CUDA OFF CACHE BOOL "Enable CUDA backend" FORCE)
  else()
    set(NEDO_CUDA ON CACHE BOOL "Enable CUDA backend" FORCE)
    set(NEDO_CUDA_REQUIRED ON)
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Arch installs the toolkit under /opt/cuda and many local CUDA installs use
# /usr/local/cuda. CMake's compiler probe primarily follows PATH, so provide
# deterministic hints without requiring users to edit shell startup files.
if(NEDO_CUDA AND NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  if(EXISTS "/opt/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

# Prefer native SASS for the GPU that will execute the wheel. This avoids
# relying on the NVIDIA driver's PTX JIT, which can reject PTX emitted by a
# newer CUDA toolkit ("unsupported toolchain"). NEDO_CUDA_ARCH is also an
# explicit escape hatch for cross-builds, e.g. NEDO_CUDA_ARCH=86 for Ampere.
if(NEDO_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{NEDO_CUDA_ARCH} AND NOT "$ENV{NEDO_CUDA_ARCH}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{NEDO_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "nedo.cpp CUDA architecture override: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    find_program(NEDO_NVIDIA_SMI nvidia-smi HINTS /usr/bin /usr/local/bin)
    if(NEDO_NVIDIA_SMI)
      execute_process(
        COMMAND "${NEDO_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
        RESULT_VARIABLE NEDO_CUDA_CC_RESULT
        OUTPUT_VARIABLE NEDO_CUDA_CC_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(NEDO_CUDA_CC_RESULT EQUAL 0)
        string(REGEX MATCH "[0-9]+\\.[0-9]+" NEDO_CUDA_CC "${NEDO_CUDA_CC_OUTPUT}")
        if(NEDO_CUDA_CC)
          string(REPLACE "." "" NEDO_CUDA_ARCH "${NEDO_CUDA_CC}")
          set(CMAKE_CUDA_ARCHITECTURES "${NEDO_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
          message(STATUS "nedo.cpp detected local CUDA architecture: sm_${NEDO_CUDA_ARCH}")
        endif()
      endif()
    endif()
  endif()
endif()

set(NEDO_CUDA_ENABLED OFF)
if(NEDO_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(NEDO_CUDA_ENABLED ON)
    message(STATUS "nedo.cpp CUDA backend: ${CMAKE_CUDA_COMPILER}")
    if(CMAKE_CUDA_ARCHITECTURES)
      message(STATUS "nedo.cpp CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  elseif(NEDO_CUDA_REQUIRED)
    message(FATAL_ERROR "NEDO_CUDA_BUILD requested but nvcc/CUDA compiler was not found")
  else()
    message(STATUS "nedo.cpp CUDA backend: unavailable; building CPU fallback")
  endif()
endif()

set(NEDO_DEVICE_SOURCE src/cuda_stub.cpp)
if(NEDO_CUDA_ENABLED)
  set(NEDO_DEVICE_SOURCE src/cuda_backend.cu)
endif()

add_library(nedo_core
  src/gguf.cpp
  src/kernels.cpp
  src/tokenizer.cpp
  src/model.cpp
  ${NEDO_DEVICE_SOURCE}
)
target_include_directories(nedo_core PUBLIC include)

if(MSVC)
  target_compile_options(nedo_core PRIVATE /O2 /Ob3 /DNOMINMAX)
  if(NEDO_NATIVE AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    target_compile_options(nedo_core PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(nedo_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-fno-math-errno;-fno-trapping-math>)
  if(NEDO_FAST_MATH)
    target_compile_options(nedo_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
  endif()
  if(NEDO_NATIVE)
    target_compile_options(nedo_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  endif()
endif()

if(NEDO_CUDA_ENABLED)
  target_compile_definitions(nedo_core PUBLIC NEDO_HAS_CUDA=1)
  target_link_libraries(nedo_core PUBLIC CUDA::cudart)
  set_target_properties(nedo_core PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
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
