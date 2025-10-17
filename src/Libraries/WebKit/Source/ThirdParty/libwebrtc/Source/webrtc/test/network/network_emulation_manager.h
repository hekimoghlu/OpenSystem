/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#ifndef TEST_NETWORK_NETWORK_EMULATION_MANAGER_H_
#define TEST_NETWORK_NETWORK_EMULATION_MANAGER_H_

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "api/field_trials_view.h"
#include "api/test/network_emulation_manager.h"
#include "api/test/simulated_network.h"
#include "api/test/time_controller.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/task_queue_for_test.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "system_wrappers/include/clock.h"
#include "test/network/cross_traffic.h"
#include "test/network/emulated_network_manager.h"
#include "test/network/emulated_turn_server.h"
#include "test/network/network_emulation.h"

namespace webrtc {
namespace test {

class NetworkEmulationManagerImpl : public NetworkEmulationManager {
 public:
  explicit NetworkEmulationManagerImpl(NetworkEmulationManagerConfig config);
  ~NetworkEmulationManagerImpl();

  EmulatedNetworkNode* CreateEmulatedNode(BuiltInNetworkBehaviorConfig config,
                                          uint64_t random_seed = 1) override;
  EmulatedNetworkNode* CreateEmulatedNode(
      std::unique_ptr<NetworkBehaviorInterface> network_behavior) override;

  SimulatedNetworkNode::Builder NodeBuilder() override;

  EmulatedEndpointImpl* CreateEndpoint(EmulatedEndpointConfig config) override;
  void EnableEndpoint(EmulatedEndpoint* endpoint) override;
  void DisableEndpoint(EmulatedEndpoint* endpoint) override;

  EmulatedRoute* CreateRoute(EmulatedEndpoint* from,
                             const std::vector<EmulatedNetworkNode*>& via_nodes,
                             EmulatedEndpoint* to) override;

  EmulatedRoute* CreateRoute(
      const std::vector<EmulatedNetworkNode*>& via_nodes) override;

  EmulatedRoute* CreateDefaultRoute(
      EmulatedEndpoint* from,
      const std::vector<EmulatedNetworkNode*>& via_nodes,
      EmulatedEndpoint* to) override;

  void ClearRoute(EmulatedRoute* route) override;

  TcpMessageRoute* CreateTcpRoute(EmulatedRoute* send_route,
                                  EmulatedRoute* ret_route) override;

  CrossTrafficRoute* CreateCrossTrafficRoute(
      const std::vector<EmulatedNetworkNode*>& via_nodes) override;

  CrossTrafficGenerator* StartCrossTraffic(
      std::unique_ptr<CrossTrafficGenerator> generator) override;
  void StopCrossTraffic(CrossTrafficGenerator* generator) override;

  EmulatedNetworkManagerInterface* CreateEmulatedNetworkManagerInterface(
      const std::vector<EmulatedEndpoint*>& endpoints) override;

  void GetStats(
      rtc::ArrayView<EmulatedEndpoint* const> endpoints,
      std::function<void(EmulatedNetworkStats)> stats_callback) override;

  void GetStats(
      rtc::ArrayView<EmulatedNetworkNode* const> nodes,
      std::function<void(EmulatedNetworkNodeStats)> stats_callback) override;

  TimeController* time_controller() override { return time_controller_.get(); }

  TimeMode time_mode() const override { return time_mode_; }

  Timestamp Now() const;

  EmulatedTURNServerInterface* CreateTURNServer(
      EmulatedTURNServerConfig config) override;

 private:
  using CrossTrafficSource =
      std::pair<std::unique_ptr<CrossTrafficGenerator>, RepeatingTaskHandle>;

  std::optional<rtc::IPAddress> GetNextIPv4Address();

  const TimeMode time_mode_;
  const EmulatedNetworkStatsGatheringMode stats_gathering_mode_;
  const std::unique_ptr<TimeController> time_controller_;
  Clock* const clock_;
  const bool fake_dtls_handshake_sizes_;
  int next_node_id_;

  RepeatingTaskHandle process_task_handle_;

  uint32_t next_ip4_address_;
  std::set<rtc::IPAddress> used_ip_addresses_;

  // All objects can be added to the manager only when it is idle.
  std::vector<std::unique_ptr<EmulatedEndpoint>> endpoints_;
  std::vector<std::unique_ptr<EmulatedNetworkNode>> network_nodes_;
  std::vector<std::unique_ptr<EmulatedRoute>> routes_;
  std::vector<std::unique_ptr<CrossTrafficRoute>> traffic_routes_;
  std::vector<CrossTrafficSource> cross_traffics_;
  std::list<std::unique_ptr<TcpMessageRouteImpl>> tcp_message_routes_;
  std::vector<std::unique_ptr<EndpointsContainer>> endpoints_containers_;
  std::vector<std::unique_ptr<EmulatedNetworkManager>> network_managers_;
  std::vector<std::unique_ptr<EmulatedTURNServer>> turn_servers_;

  std::map<EmulatedEndpoint*, EmulatedNetworkManager*>
      endpoint_to_network_manager_;

  // Must be the last field, so it will be deleted first, because tasks
  // in the TaskQueue can access other fields of the instance of this class.
  TaskQueueForTest task_queue_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_NETWORK_NETWORK_EMULATION_MANAGER_H_
