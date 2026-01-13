/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "smem.h"
#include "smem_shm.h"
#include "smem_trans.h"

namespace xllm {
namespace layer {

class MfWeightTransfer {
 public:
  MfWeightTransfer(int device_id,
                   int rank_id,
                   int rank_size,
                   const std::string& ip_port,
                   const std::string& session_id,
                   smem_trans_role_t role)
      : device_id_(device_id),
        rank_id_(rank_id),
        rank_size_(rank_size),
        ip_port_(ip_port),
        session_id_(session_id) {
    const uint32_t LOG_LEVEL_WARNING = 2;
    smem_set_log_level(LOG_LEVEL_WARNING);
    auto ret = smem_init(0);
    CHECK_EQ(ret, 0) << "smem init failed, ret:" << ret
                     << ", rank:" << rank_id_;
    if (rank_id_ == 0) {
      ret = smem_create_config_store(ip_port_.c_str());
      CHECK_EQ(ret, 0) << "smem create config store failed, ret:" << ret
                       << ", rank:" << rank_id_;
    }

    smem_trans_config_init(&trans_config_);
    trans_config_.role = role;
    trans_config_.deviceId = device_id_;
    trans_config_.dataOpType = SMEMB_DATA_OP_SDMA;

    ret = smem_trans_init(&trans_config_);
    CHECK_EQ(ret, 0) << "smem trans init failed, ret:" << ret
                     << ", rank:" << rank_id_;

    trans_handle_ = smem_trans_create(
        ip_port_.c_str(), session_id_.c_str(), &trans_config_);
    CHECK(trans_handle_ != nullptr)
        << "smem trans create failed, rank:" << rank_id_;

    (void)smem_shm_config_init(&shm_config_);
    shm_config_.startConfigStoreServer = false;

    ret = smem_shm_init(
        ip_port_.c_str(), rank_size_, rank_id_, device_id_, &shm_config_);
    CHECK_EQ(ret, 0) << "smem shm init failed, ret:" << ret
                     << ", rank:" << rank_id_;

    shm_handle_ = smem_shm_create(0,
                                  rank_size_,
                                  rank_id_,
                                  1024ULL * 1024 * 2,
                                  SMEMS_DATA_OP_MTE,
                                  0,
                                  &shm_gva_);
    CHECK(shm_handle_ != nullptr)
        << "smem shm create failed, rank:" << rank_id_;

    LOG(INFO) << "mf_weight_transfer init done, rank:" << rank_id_
              << ", session_id:" << session_id_;
    barrier();
    LOG(INFO) << "mf_weight_transfer init barrier passed, rank:" << rank_id_;
  }

  ~MfWeightTransfer() {
    smem_shm_destroy(shm_handle_, 0);
    smem_shm_uninit(0);
    smem_trans_destroy(trans_handle_, 0);
    smem_trans_uninit(0);
    smem_uninit();
  }

  void init(std::vector<void*>& weight_addrs,
            std::vector<size_t>& weight_sizes) {
    LOG(INFO) << "mf_weight_transfer init start, rank:" << rank_id_;
    weight_addrs_ = std::move(weight_addrs);
    weight_sizes_ = std::move(weight_sizes);

    gather_peer_addrs();
    register_weight();
    barrier();
    LOG(INFO) << "mf_weight_transfer init done, rank:" << rank_id_;
  }

  void gather_peer_addrs() {
    LOG(INFO) << "mf_weight_transfer gather_peer_addrs start, rank:"
              << rank_id_;
    global_weight_addrs_.resize(rank_size_);
    for (int i = 0; i < rank_size_; ++i) {
      global_weight_addrs_[i].resize(weight_sizes_.size());
    }
    if (weight_addrs_.empty()) {
      LOG(WARNING) << "mf_weight_transfer gather_peer_addrs skip: empty";
      barrier();
      return;
    }

    const size_t addr_count = weight_addrs_.size();
    std::vector<void*> weight_addrs_gather(rank_size_ * addr_count);
    int ret = smem_shm_control_allgather(
        shm_handle_,
        reinterpret_cast<char*>(weight_addrs_.data()),
        sizeof(void*) * addr_count,
        reinterpret_cast<char*>(weight_addrs_gather.data()),
        sizeof(void*) * addr_count * rank_size_);
    CHECK_EQ(ret, 0) << "smem shm control allgather failed, ret:" << ret
                     << ", rank:" << rank_id_;
    barrier();
    for (int j = 0; j < rank_size_; ++j) {
      for (size_t i = 0; i < addr_count; ++i) {
        global_weight_addrs_[j][i] = weight_addrs_gather[j * addr_count + i];
      }
    }
    LOG(INFO) << "mf_weight_transfer allgather done, rank:" << rank_id_
              << ", count:" << addr_count;
    LOG(INFO) << "mf_weight_transfer gather_peer_addrs done, rank:" << rank_id_;
  }

  void register_weight() {
    LOG(INFO) << "mf_weight_transfer register_weight start, rank:" << rank_id_;
    int ret = smem_trans_batch_register_mem(
        trans_handle_,
        weight_addrs_.data(),
        weight_sizes_.data(),
        static_cast<uint32_t>(weight_sizes_.size()),
        0);
    CHECK_EQ(ret, 0) << "smem trans batch register weight mem failed, ret:"
                     << ret << ", rank:" << rank_id_;
    std::this_thread::sleep_for(std::chrono::seconds(10UL));
    LOG(INFO) << "mf_weight_transfer register_weight done, rank:" << rank_id_;
  }

  void barrier() {
    LOG(INFO) << "mf_weight_transfer barrier enter, rank:" << rank_id_;
    auto ret = smem_shm_control_barrier(shm_handle_);
    CHECK_EQ(ret, 0) << "smem shm control barrier failed, ret:" << ret
                     << ", rank:" << rank_id_;
    LOG(INFO) << "mf_weight_transfer barrier exit, rank:" << rank_id_;
  }

  void transfer_weight() {
    LOG(ERROR) << "transfer_weight is not implemented yet.";
  }

 private:
  int device_id_;
  int rank_id_;
  int rank_size_;
  std::string ip_port_;
  smem_trans_config_t trans_config_;
  smem_trans_t trans_handle_;
  std::string session_id_;

  smem_shm_config_t shm_config_;
  smem_shm_t shm_handle_;
  void* shm_gva_ = nullptr;

  std::vector<void*> weight_addrs_;
  std::vector<std::vector<void*>> global_weight_addrs_;
  std::vector<size_t> weight_sizes_;
};

}  // namespace layer
}  // namespace xllm
