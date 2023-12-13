#pragma once

#include <c10/xpu/XPUException.h>
#include <c10/xpu/XPUMacros.h>
#include <c10/xpu/impl/XPUDeviceAttributes.h>

namespace c10::xpu {

C10_XPU_API void xpuGetDeviceCount(int* device_count);

C10_XPU_API void xpuGetDevice(int* cur_device);

C10_XPU_API void xpuSetDevice(int device_id);

C10_XPU_API sycl::device& xpuGetRawDevice(int device_id);

C10_XPU_API sycl::context& xpuGetDeviceContext();

C10_XPU_API void xpuGetDeviceProperties(
    xpuDeviceProp* device_prop,
    int device_id);

C10_XPU_API void xpuPointerGetDevice(xpuPointerAttributes* attr, void* ptr);

C10_XPU_API XPU_STATUS xpuPrefetchDeviceCount(int* device_count);

} // namespace c10::xpu
