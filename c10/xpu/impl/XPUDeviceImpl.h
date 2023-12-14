#pragma once

#include <c10/xpu/XPUMacros.h>
#include <c10/xpu/impl/XPUDeviceAttributes.h>

namespace c10::xpu {

void xpuGetDeviceCount(int* device_count);

void xpuGetDevice(int* cur_device);

void xpuSetDevice(int device_id);

C10_XPU_API sycl::device& xpuGetRawDevice(int device_id);

C10_XPU_API sycl::context& xpuGetDeviceContext();

C10_XPU_API void xpuGetDeviceProperties(
    xpuDeviceProp* device_prop,
    int device_id);

C10_XPU_API void xpuPointerGetDevice(xpuPointerAttributes* attr, void* ptr);

// *device_count is valid only when this function returns true.
bool xpuPrefetchDeviceCount(int* device_count);

} // namespace c10::xpu
