# ---[ xpu

# Poor man's include guard
if(TARGET torch::xpurt)
  return()
endif()

# Find Intel SYCL Library.
find_package(IntelSYCL REQUIRED)
if(NOT IntelSYCL_FOUND)
  set(CAFFE2_USE_XPU OFF)
  return()
endif()

# Intel SYCL library interface
add_library(XPU::sycl INTERFACE IMPORTED)

set_property(
    TARGET XPU::sycl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${SYCL_INCLUDE_DIR})
set_property(
    TARGET XPU::sycl PROPERTY INTERFACE_LINK_LIBRARIES
    ${SYCL_LIBRARY})

# xpu
add_library(caffe2::xpu INTERFACE IMPORTED)
set_property(
    TARGET caffe2::xpu PROPERTY INTERFACE_LINK_LIBRARIES
    XPU::sycl)

# xpurt
add_library(torch::xpurt INTERFACE IMPORTED)
set_property(
    TARGET torch::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    XPU::sycl)
