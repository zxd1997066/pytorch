#include <c10/xpu/XPUFunctions.h>

namespace c10::xpu {

namespace {
static inline int device_count_impl() {
  int count = 0;
  xpuGetDeviceCount(&count);
  return count;
}
} // anonymous namespace

DeviceIndex device_count() {
  // initialize number of devices only once
  static int count = []() { return device_count_impl(); }();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  auto count = device_count();
  // Zero gpus could produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No XPU GPUs are available");
  return count;
}

DeviceIndex current_device() {
  int cur_device = 0;
  xpuGetDevice(&cur_device);
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  xpuSetDevice(static_cast<int>(device));
}

DeviceIndex prefetch_device_count() {
  int count = 0;
  auto status = xpuPrefetchDeviceCount(&count);
  if (status != XPU_SUCCESS) {
    return -1;
  }
  return static_cast<DeviceIndex>(count);
}

int ExchangeDevice(int to_device) {
  int cur_device = 0;
  xpuGetDevice(&cur_device);
  if (to_device == cur_device) {
    return cur_device;
  }
  xpuSetDevice(to_device);
  return cur_device;
}

int MaybeExchangeDevice(int to_device) {
  return c10::xpu::ExchangeDevice(to_device);
}

} // namespace c10::xpu
