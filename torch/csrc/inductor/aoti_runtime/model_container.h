#pragma once

#include <algorithm>
#include <deque>
#include <future>
#include <mutex>
#include <shared_mutex>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model.h>

namespace torch {
namespace aot_inductor {

class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(
      size_t num_models,
      bool is_cpu = false,
      std::optional<std::string> cubin_dir = std::nullopt) {
    constants_map_ = std::make_shared<ConstantMap>();
    constants_array_ = std::make_shared<std::vector<AtenTensorHandle>>();
    use_secondary = false;
    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(
          constants_map_, constants_array_, cubin_dir));
      available_models_.push_back(models_.back().get());
    }

    // Note that the all following fields (input_names_, output_names,
    // etc) can be filled in by the AOT
    // codegen. However, we choose to query such information from
    // the owned AOTInductorModel for a couple of reasons:
    //   * simplify the codegen templates
    //   * reduce information fragmentation and duplication
    //   * the initialization process below is done only once when the container
    //     is constructed, so it would have little performance impact
    auto* model = available_models_[0];
    size_t num_inputs = model->num_inputs();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_.push_back(model->input_name(i));
    }

    size_t num_outputs = model->num_outputs();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.push_back(model->output_name(i));
    }

    model->load_constants(is_cpu);
#ifdef USE_CUDA
    constant_blob_ = model->release_constant_blob();
    constants_internal_offset_.resize(model->num_constants());
    model->compute_cuda_constant_blob(blob_size_, constants_internal_offset_);
#endif

    for (auto& model : models_) {
      model->update_constants_map(constants_map_);
    }

    in_spec_ = model->get_in_spec();
    out_spec_ = model->get_out_spec();
  }

  void run(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    auto* model = get_available_model();
    try {
      model->run(input_handles, output_handles, stream, proxy_executor);
    } catch (...) {
      std::lock_guard lk(models_mutex_);
      available_models_.push_back(model);
      throw;
    }

    {
      std::lock_guard lk(models_mutex_);
      pending_models_.push_back(model);
    }
    pending_models_available_.notify_one();
  }

  // This function updates the inactive buffer for storing constants.
  // It will update the buffer, the mapping and the array mapping.
  // We can later change the inactive buffer to active with corresponding
  // function calls (TODO.)
  void update_inactive_constants_buffer(
      const std::unordered_map<std::string, AtenTensorHandle>& constants_map) {
#ifdef USE_CUDA
    if (this->num_models() == 0) {
      throw std::runtime_error("No model available in container!");
    }
    auto num_constants = models_[0]->num_constants();

    // auto constants_map = get_inactive_map();
    // auto constants_array = get_inactive_array();
    auto constants_blob = get_inactive_blob();
    auto* constants_blob_ptr = static_cast<uint8_t*>(constant_blob.get());

    auto constants_map = get_inactive_map();

    for (size_t idx = 0; idx < num_constants; idx++) {
      auto constant_name = std::string(models_[0]->constant_name(idx));
      auto it = constants_map.find(constant_name);
      if (it == constants_map.end()) {
        throw std::runtime_error(
            std::string("Cannot find constants ") + constant_name +
            std::string(" in constants_map!"));
      }

      // Move the data to container handled blob.
      auto* internal_constants_ptr =
          constants_blob_ptr + constants_internal_offset_[idx];
      void* user_constant_ptr;
      int64_t* constant_size;
      aoti_torch_get_data_ptr(it->second, &user_constant_ptr);
      aoti_torch_get_sizes(it->second, &constant_size);

      AOTI_RUNTIME_DEVICE_CHECK(cudaMemcpy(
          internal_constants_ptr,
          user_constant_ptr,
          *constant_size,
          cudaMemcpyDeviceToDevice));

      // Generate Tensor from container handled blob.
      // We extract stride and offset from provided Tensor since we do not
      // guarantee that the tensor is contiguous.
      AtenTensorHandle tensor_handle;
      int64_t* stride;
      size_t offset;
      int device_idx = -1;
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(it->second, &stride));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_storage_offset(it->second, &offset));
      AOTI_RUNTIME_DEVICE_CHECK(cudaGetDevice(&device_idx));
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
          internal_constants_ptr,
          model_[0]->constant_ndim(idx),
          model_[0]->constant_shape(idx),
          stride,
          offset,
          model_[0]->constant_type(idx),
          aoti_torch_device_type_cuda(),
          device_idx,
          &tensor_handle));

      // Now place the tensor to constants_map. Note at this point the ownership
      // of the tensor_handle will be taken over.
      constants_map->emplace(constant_name, tensor_handle);
    }

    // Update the inactive constant array.
    update_array_from_map(get_inactive_array(), constants_map);
#endif // USE_CUDA
  }

  void update_array_from_map(
      std::shared_ptr<std::vector<AtenTensorHandle>> constants_array,
      std::shared_ptr<ConstantMap> constants_map) {
    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      constants_array->at(idx) =
          constants_map->find(models_[0]->constant_name(idx))->second;
    }
  }

  size_t num_inputs() const {
    return input_names_.size();
  }

  size_t num_outputs() const {
    return output_names_.size();
  }

  const char* input_name(size_t idx) const {
    return input_names_.at(idx).c_str();
  }

  const char* output_name(size_t idx) const {
    return output_names_.at(idx).c_str();
  }

  size_t num_models() const {
    return models_.size();
  }

  const char* get_in_spec() const {
    return in_spec_;
  }

  const char* get_out_spec() const {
    return out_spec_;
  }

 private:
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  const char* in_spec_;
  const char* out_spec_;

#ifdef USE_CUDA
  // Holds the blob storage for constants' at::Tensor for CUDA.
  CUDAPtr constant_blob_;
  CUDAPtr constant_blob_secondary_;

  // Let's place this within USE_CUDA at the moment before we fully support
  // update for CPU cases.
  bool use_secondary;
  size_t blob_size_;
  std::vector<size_t> constants_internal_offset_;
#endif // USE_CUDA

  // Holds the mapping of constants to at::Tensor.
  // The underlying data of at::Tensor is in either constant_blob_ (for CUDA).
  // or _binary_constants_bin_start (for CPU).
  std::shared_ptr<ConstantMap> constants_map_;
  std::shared_ptr<ConstantMap> constants_map_secondary_;

  // Holds the indexed array of constant for faster lookup during runtime.
  std::shared_ptr<std::vector<AtenTensorHandle>> constants_array_;
  std::shared_ptr<std::vector<AtenTensorHandle>> constants_array_secondary_;

  // Holds all the AOTInductorModel instances owned by this container.
  std::vector<std::unique_ptr<AOTInductorModel>> models_;

  // Holds the AOTInductorModel instances available for inference.
  std::vector<AOTInductorModel*> available_models_;

  // Holds the AOTInductorModel instances that have started running
  // inference and can be placed onto available_models_ upon their
  // completion.
  std::deque<AOTInductorModel*> pending_models_;

  // Protects available_models_ and pending_models_.
  std::mutex models_mutex_;

  // Notified whenever a model is placed onto pending_models_.
  std::condition_variable pending_models_available_;

  AOTInductorModel* get_available_model() {
    std::unique_lock lk(models_mutex_);
    if (available_models_.empty()) {
      reclaim_finished_models(lk);
    }
    auto* result = available_models_.back();
    available_models_.pop_back();
    return result;
  }

#ifdef USE_CUDA
  CUDAPtr get_inactive_blob() {
    if (use_secondary) {
      return constant_blob_;
    } else {
      if (!constant_blob_secondary_) {
        constant_blob_secondary_ = RAII_cudaMalloc(blob_size_);
      }
      return constant_blob_secondary_;
    }
  }
#endif // USE_CUDA

  std::shared_ptr<ConstantMap> get_inactive_map() {
    if (use_secondary) {
      return constants_map_;
    } else {
      if (!constants_map_secondary_) {
        constants_map_secondary_ = std::make_shared<ConstantMap>();
      }
      return constants_map_secondary_;
    }
  }

  std::shared_ptr<std::vector<AtenTensorHandle>> get_inactive_array() {
    if (use_secondary) {
      return constants_array_;
    } else {
      if (!constants_array_secondary_) {
        constants_array_secondary_ =
            std::make_shared<std::vector<AtenTensorHandle>>(
                models_[0]->num_constants());
      }
      return constants_array_secondary_;
    }
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
    // push finished model instances to the end of pending_models_
    auto it = std::stable_partition(
        pending_models_.begin(),
        pending_models_.end(),
        [](AOTInductorModel* m) { return !m->is_finished(); });

    if (it != pending_models_.end()) {
      // We have finished model instances that can be pushed into
      // available_models_ so that we don't have to be blocked on waiting
      // the pending_models_available_ condition.
      available_models_.insert(
          available_models_.end(), it, pending_models_.end());
      pending_models_.erase(it, pending_models_.end());
      return;
    }

    pending_models_available_.wait(
        lk, [this]() { return !pending_models_.empty(); });
    // Let's make the schedule simple first. We always wait on the first
    // pending_models_ to be complete.
    auto* model = pending_models_.front();
    pending_models_.pop_front();
    lk.unlock();
    try {
      model->wait_for_completion();
    } catch (...) {
      lk.lock();
      available_models_.push_back(model);
      throw;
    }
    lk.lock();
    available_models_.push_back(model);
  }
};

} // namespace aot_inductor
} // namespace torch
