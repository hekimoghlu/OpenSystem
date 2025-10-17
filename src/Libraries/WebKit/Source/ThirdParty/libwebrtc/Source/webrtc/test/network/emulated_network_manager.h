/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef TEST_NETWORK_EMULATED_NETWORK_MANAGER_H_
#define TEST_NETWORK_EMULATED_NETWORK_MANAGER_H_

#include <functional>
#include <memory>
#include <vector>

#include "api/sequence_checker.h"
#include "api/test/network_emulation_manager.h"
#include "api/test/time_controller.h"
#include "rtc_base/ip_address.h"
#include "rtc_base/network.h"
#include "rtc_base/socket_server.h"
#include "rtc_base/thread.h"
#include "test/network/network_emulation.h"

namespace webrtc {
namespace test {

// Framework assumes that rtc::NetworkManager is called from network thread.
class EmulatedNetworkManager : public rtc::NetworkManagerBase,
                               public sigslot::has_slots<>,
                               public EmulatedNetworkManagerInterface {
 public:
  EmulatedNetworkManager(TimeController* time_controller,
                         TaskQueueForTest* task_queue,
                         EndpointsContainer* endpoints_container);

  void EnableEndpoint(EmulatedEndpointImpl* endpoint);
  void DisableEndpoint(EmulatedEndpointImpl* endpoint);

  // NetworkManager interface. All these methods are supposed to be called from
  // the same thread.
  void StartUpdating() override;
  void StopUpdating() override;

  // We don't support any address interfaces in the network emulation framework.
  std::vector<const rtc::Network*> GetAnyAddressNetworks() override {
    return {};
  }

  // EmulatedNetworkManagerInterface API
  rtc::Thread* network_thread() override { return network_thread_.get(); }
  rtc::NetworkManager* network_manager() override { return this; }
  rtc::PacketSocketFactory* packet_socket_factory() override {
    return packet_socket_factory_.get();
  }
  std::vector<EmulatedEndpoint*> endpoints() const override {
    return endpoints_container_->GetEndpoints();
  }
  void GetStats(
      std::function<void(EmulatedNetworkStats)> stats_callback) const override;

 private:
  void UpdateNetworksOnce();
  void MaybeSignalNetworksChanged();

  TaskQueueForTest* const task_queue_;
  const EndpointsContainer* const endpoints_container_;
  // The `network_thread_` must outlive `packet_socket_factory_`, because they
  // both refer to a socket server that is owned by `network_thread_`. Both
  // pointers are assigned only in the constructor, but the way they are
  // initialized unfortunately doesn't work with const std::unique_ptr<...>.
  std::unique_ptr<rtc::Thread> network_thread_;
  std::unique_ptr<rtc::PacketSocketFactory> packet_socket_factory_;
  bool sent_first_update_ RTC_GUARDED_BY(network_thread_);
  int start_count_ RTC_GUARDED_BY(network_thread_);
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_NETWORK_EMULATED_NETWORK_MANAGER_H_
