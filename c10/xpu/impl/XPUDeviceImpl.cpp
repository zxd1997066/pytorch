#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/xpu/impl/XPUDeviceImpl.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <cmath>
#include <deque>
#include <mutex>
#include <vector>

namespace c10::xpu {
namespace {

/*
 * Note [Device Management]
 *
 * Intel GPU device is based on sycl::device enumerated via SYCL runtime and
 * device runtime status can be managed via a sycl device pool. The number of
 * GPU devices is determined at run time.
 *
 * Currently, there is one SYCL device pool and the device pool is lazily
 * created only once. The device management mechanism is thread local safe.
 * The same default sycl context can shared for each sycl device.
 *
 * Device properties are initialized via the specific raw device.
 */
static c10::once_flag init_flag;
static thread_local int curDeviceIndex = 0;

struct XPUDevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::vector<std::unique_ptr<sycl::context>> contexts;
} gDevicePool;

static void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from GPU platform firstly.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    // Enumerated GPU devices from platform.
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        devices.push_back(std::make_unique<sycl::device>(device));
      }
    }
  }
}

static inline int DeviceCountImpl(
    std::vector<std::unique_ptr<sycl::device>>& devices) {
  enumDevices(devices);
  return static_cast<int>(devices.size());
}

static inline void initGlobalDevicePoolState() {
  auto device_count = DeviceCountImpl(gDevicePool.devices);
  if (device_count <= 0) {
    TORCH_WARN("XPU device count is zero!");
  }

  // Here we use default context provided by Intel oneapi extension for each
  // Intel GPU device. So the size of contexts is 1.
  gDevicePool.contexts.resize(1);
  gDevicePool.contexts[0] = std::make_unique<sycl::context>(
      gDevicePool.devices[0]->get_platform().ext_oneapi_get_default_context());
}

static inline void initDevicePoolCallOnce() {
  c10::call_once(init_flag, initGlobalDevicePoolState);
}

static void initDeviceProperties(xpuDeviceProp* device_prop, int device_id) {
  using namespace sycl::info;
  using namespace sycl::ext;
  // Get raw sycl device associated with device_id.
  auto& device = *gDevicePool.devices[device_id];

  // clang-format off
  // Initialize the device properties associated with the specific device.
  device_prop->device_name = device.get_info<device::name>();
  device_prop->device_type = device.get_info<device::device_type>();
  device_prop->platform_name = device.get_info<device::platform>().get_info<platform::name>();
  device_prop->vendor = device.get_info<device::vendor>();
  device_prop->driver_version = device.get_info<device::driver_version>();
  device_prop->is_available = device.get_info<device::is_available>();
  device_prop->max_param_size = device.get_info<device::max_parameter_size>();
  device_prop->max_compute_units = device.get_info<device::max_compute_units>();
  device_prop->max_work_item_dims = device.get_info<device::max_work_item_dimensions>();
  device_prop->max_work_group_size = device.get_info<device::max_work_group_size>();
  device_prop->max_num_sub_groups = device.get_info<device::max_num_sub_groups>();
  device_prop->sub_group_sizes = device.get_info<device::sub_group_sizes>();
  device_prop->max_clock_freq = device.get_info<device::max_clock_frequency>();
  device_prop->address_bits = device.get_info<device::address_bits>();
  device_prop->max_mem_alloc_size = device.get_info<device::max_mem_alloc_size>();
  device_prop->mem_base_addr_align = device.get_info<device::mem_base_addr_align>();
  device_prop->half_fp_config = device.get_info<device::half_fp_config>();
  device_prop->single_fp_config = device.get_info<device::single_fp_config>();
  device_prop->double_fp_config = device.get_info<device::double_fp_config>();
  device_prop->global_mem_size = device.get_info<device::global_mem_size>();
  device_prop->global_mem_cache_type = device.get_info<device::global_mem_cache_type>();
  device_prop->global_mem_cache_size = device.get_info<device::global_mem_cache_size>();
  device_prop->global_mem_cache_line_size = device.get_info<device::global_mem_cache_line_size>();
  device_prop->local_mem_type = device.get_info<device::local_mem_type>();
  device_prop->local_mem_size = device.get_info<device::local_mem_size>();
  device_prop->max_sub_devices = device.get_info<device::partition_max_sub_devices>();
  device_prop->profiling_resolution = device.get_info<device::profiling_timer_resolution>();
  device_prop->pref_vec_width_char = device.get_info<device::preferred_vector_width_char>();
  device_prop->pref_vec_width_short = device.get_info<device::preferred_vector_width_short>();
  device_prop->pref_vec_width_int = device.get_info<device::preferred_vector_width_int>();
  device_prop->pref_vec_width_long = device.get_info<device::preferred_vector_width_long>();
  device_prop->pref_vec_width_float = device.get_info<device::preferred_vector_width_float>();
  device_prop->pref_vec_width_double = device.get_info<device::preferred_vector_width_double>();
  device_prop->pref_vec_width_half = device.get_info<device::preferred_vector_width_half>();
  device_prop->native_vec_width_char = device.get_info<device::native_vector_width_char>();
  device_prop->native_vec_width_short = device.get_info<device::native_vector_width_short>();
  device_prop->native_vec_width_int = device.get_info<device::native_vector_width_int>();
  device_prop->native_vec_width_long = device.get_info<device::native_vector_width_long>();
  device_prop->native_vec_width_float = device.get_info<device::native_vector_width_float>();
  device_prop->native_vec_width_double = device.get_info<device::native_vector_width_double>();
  device_prop->native_vec_width_half = device.get_info<device::native_vector_width_half>();
  // Intel extension
  device_prop->gpu_eu_count = device.has(sycl::aspect::ext_intel_gpu_eu_count)
      ? device.get_info<intel::info::device::gpu_eu_count>()
      : 512;
  device_prop->gpu_eu_count_per_subslice = device.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)
      ? device.get_info<intel::info::device::gpu_eu_count_per_subslice>()
      : 8;
  device_prop->gpu_eu_simd_width = device.has(sycl::aspect::ext_intel_gpu_eu_simd_width)
      ? device.get_info<intel::info::device::gpu_eu_simd_width>()
      : 8;
  device_prop->gpu_hw_threads_per_eu = device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)
      ? device.get_info<intel::info::device::gpu_hw_threads_per_eu>()
      : 8;
  device_prop->support_fp64 = device.has(sycl::aspect::fp64);
  device_prop->support_atomic64 = device.has(sycl::aspect::atomic64);
  // clang-format on
  return;
}

} // anonymous namespace

void xpuGetDeviceCount(int* device_count) {
  initDevicePoolCallOnce();
  *device_count = static_cast<int>(gDevicePool.devices.size());
  return;
}

void xpuGetDevice(int* cur_device) {
  initDevicePoolCallOnce();
  *cur_device = curDeviceIndex;
  return;
}

void xpuSetDevice(int device_id) {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      device_id < static_cast<int>(gDevicePool.devices.size()),
      "xpuSetDevice: device_id is out of range.");
  curDeviceIndex = device_id;
  return;
}

sycl::device& xpuGetRawDevice(int device_id) {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      device_id < static_cast<int>(gDevicePool.devices.size()),
      "xpuGetRawDevice: device_id is out of range.");
  return *gDevicePool.devices[device_id];
}

sycl::context& xpuGetDeviceContext() {
  initDevicePoolCallOnce();
  return *gDevicePool.contexts[0];
}

void xpuGetDeviceProperties(xpuDeviceProp* device_prop, int device_id) {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      device_prop,
      "xpuGetDeviceProperties: device_prop is an invalid pointer.");
  TORCH_CHECK(
      device_id < static_cast<int>(gDevicePool.devices.size()),
      "xpuGetDeviceProperties: device_id is out of range.");
  initDeviceProperties(device_prop, device_id);
}

void xpuPointerGetDevice(xpuPointerAttributes* attr, void* ptr) {
  initDevicePoolCallOnce();
  TORCH_CHECK(ptr, "xpuPointerGetDevice: ptr is an invalid pointer.");
  TORCH_CHECK(attr, "xpuPointerGetDevice: attr is an invalid pointer.");
  attr->type = sycl::get_pointer_type(ptr, xpuGetDeviceContext());
  attr->device = -1;
  if (attr->type != sycl::usm::alloc::device) {
    return;
  }
  sycl::device raw_device =
      sycl::get_pointer_device(ptr, xpuGetDeviceContext());
  auto match_device = [raw_device](const auto& device) -> bool {
    return raw_device == *device;
  };
  auto it = std::find_if(
      gDevicePool.devices.begin(), gDevicePool.devices.end(), match_device);
  if (it != gDevicePool.devices.end()) {
    attr->device =
        static_cast<int>(std::distance(gDevicePool.devices.begin(), it));
    return;
  }
  TORCH_CHECK(false, "Cant't find the pointer from XPU devices.");
}

/*
 * Note [Runtime in Multiprocessing]
 *
 * We have known the limitation of fork support in SYCL runtime. If we call
 * runtime APIs in parent process, then fork a child process, there will be an
 * error in runtime if submit any kernel in parent process or child process.
 *
 * In general, SYCL runtime initialization must be called after fork, not
 * before. So we have to call runtime APIs using another fork, pipe the result
 * back to the parent, and then fork the actual child process.
 *
 * We have to fork another child process first. Then query device count using
 * SYCL runtime APIs. Finally pipe the result to parent process. Now we can
 * check if XPU device is available and fork the actual child process to do the
 * calculation.
 */

// This function can be used to get device count and no exception. It is used in
// device_count() and is_avaialble() such that both two functions can be called
// before forking process.
bool xpuPrefetchDeviceCount(int* device_count) {
#ifndef _WIN32
  std::array<int, 1> buffer;
  std::array<int, 2> pipefd;
  if (pipe(pipefd.data()) != 0) {
    return false;
  }

  // See Note [Runtime in Multiprocessing].
  int pid = fork();
  if (pid < 0) {
    return false;
  } else if (pid == 0) { // child process
    std::vector<std::unique_ptr<sycl::device>> devices;
    buffer[0] = DeviceCountImpl(devices);
    close(pipefd[0]);
    write(pipefd[1], buffer.data(), sizeof(buffer));
    close(pipefd[1]);
    _exit(0);
  } else { // parent process
    wait(NULL);
    close(pipefd[1]);
    read(pipefd[0], buffer.data(), sizeof(buffer));
    close(pipefd[0]);
  }

  *device_count = buffer[0];
  return true;
#else
  return false;
#endif
}

} // namespace c10::xpu
