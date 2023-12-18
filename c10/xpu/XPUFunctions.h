#pragma once

#include <c10/core/Device.h>
#include <c10/xpu/XPUDeviceProp.h>
#include <c10/xpu/XPUMacros.h>

// The naming convention used here matches the naming convention of torch.xpu

namespace c10::xpu {

// Log a warning only once if no devices are detected.
C10_XPU_API DeviceIndex device_count();

// Throws an error if no devices are detected.
C10_XPU_API DeviceIndex device_count_ensure_non_zero();

// If this function fails, return -1. Otherwise, return the number of Intel GPUs
// without the limitation of SYCL runtime in multi-processing.
C10_XPU_API DeviceIndex prefetch_device_count();

C10_XPU_API DeviceIndex current_device();

C10_XPU_API void set_device(DeviceIndex device);

C10_XPU_API int ExchangeDevice(int device);

C10_XPU_API int MaybeExchangeDevice(int to_device);

C10_XPU_API sycl::device& xpuGetRawDevice(int device);

C10_XPU_API sycl::context& xpuGetDeviceContext();

C10_XPU_API void xpuGetDeviceProperties(xpuDeviceProp* device_prop, int device);

C10_XPU_API int xpuPointerGetDevice(void* ptr);

} // namespace c10::xpu
