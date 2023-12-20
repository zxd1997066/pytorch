# ---[ xpu

# Poor man's include guard
if(TARGET torch::xpurt)
  return()
endif()

# Find SYCL library.
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  set(CAFFE2_USE_XPU OFF)
  return()
endif()

# SYCL library interface
add_library(torch::sycl INTERFACE IMPORTED)

set_property(
    TARGET torch::sycl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${SYCL_INCLUDE_DIR})
set_property(
    TARGET torch::sycl PROPERTY INTERFACE_LINK_LIBRARIES
    ${SYCL_LIBRARY})

# xpu
add_library(caffe2::xpu INTERFACE IMPORTED)
set_property(
    TARGET caffe2::xpu PROPERTY INTERFACE_LINK_LIBRARIES
    torch::sycl)

# xpurt
add_library(torch::xpurt INTERFACE IMPORTED)
set_property(
    TARGET torch::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    torch::sycl)
