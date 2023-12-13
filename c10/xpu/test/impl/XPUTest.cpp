#include <gtest/gtest.h>

#include <c10/xpu/XPUFunctions.h>

#define ASSERT_EQ_XPU(X, Y) \
  {                         \
    bool _isEQ = X == Y;    \
    ASSERT_TRUE(_isEQ);     \
  }

bool has_xpu() {
  int count = 0;
  c10::xpu::xpuGetDeviceCount(&count);
  return count != 0;
}

TEST(XPUTest, DeviceCount) {
#ifndef _WIN32
  ASSERT_EQ_XPU(c10::xpu::device_count(), c10::xpu::prefetch_device_count());
#endif
  return;
}

TEST(XPUTest, DeviceBehavior) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::set_device(0);
  ASSERT_EQ_XPU(c10::xpu::current_device(), 0);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  c10::xpu::set_device(1);
  ASSERT_EQ_XPU(c10::xpu::current_device(), 1);
  ASSERT_EQ_XPU(c10::xpu::ExchangeDevice(0), 1);
  ASSERT_EQ_XPU(c10::xpu::current_device(), 0);
}

TEST(XPUTest, DeviceProperties) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::xpuDeviceProp device_prop{};
  c10::xpu::xpuGetDeviceProperties(&device_prop, 0);

  ASSERT_TRUE(device_prop.gpu_eu_count > 0);
}

TEST(XPUTest, PointerGetDevice) {
  if (!has_xpu()) {
    return;
  }

  sycl::device& raw_device = c10::xpu::xpuGetRawDevice(0);
  void* ptr =
      sycl::malloc_device(8, raw_device, c10::xpu::xpuGetDeviceContext());
  c10::xpu::xpuPointerAttributes attr{};
  c10::xpu::xpuPointerGetDevice(&attr, ptr);
  ASSERT_EQ_XPU(attr.type, sycl::usm::alloc::device);
  ASSERT_EQ_XPU(attr.device, 0);
  sycl::free(ptr, c10::xpu::xpuGetDeviceContext());

  int dummy = 0;
  c10::xpu::xpuPointerGetDevice(&attr, &dummy);
  ASSERT_EQ_XPU(attr.type, sycl::usm::alloc::unknown);
}
