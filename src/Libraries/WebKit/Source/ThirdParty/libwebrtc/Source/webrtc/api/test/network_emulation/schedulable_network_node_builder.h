/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#ifndef API_TEST_NETWORK_EMULATION_SCHEDULABLE_NETWORK_NODE_BUILDER_H_
#define API_TEST_NETWORK_EMULATION_SCHEDULABLE_NETWORK_NODE_BUILDER_H_

#include <cstdint>
#include <optional>

#include "absl/functional/any_invocable.h"
#include "api/test/network_emulation/network_config_schedule.pb.h"
#include "api/test/network_emulation_manager.h"
#include "api/units/timestamp.h"

namespace webrtc {

class SchedulableNetworkNodeBuilder {
 public:
  SchedulableNetworkNodeBuilder(
      webrtc::NetworkEmulationManager& net,
      network_behaviour::NetworkConfigSchedule schedule);
  // set_start_condition allows a test to control when the schedule start.
  // `start_condition` is invoked every time a packet is enqueued on the network
  // until the first time `start_condition` returns true. Until then, the first
  // NetworkConfigScheduleItem is used. There is no guarantee on which
  // thread/task queue that will be used.
  void set_start_condition(
      absl::AnyInvocable<bool(webrtc::Timestamp)> start_condition);

  // If no random seed is provided, one will be created.
  // The random seed is required for loss rate and to delay standard deviation.
  webrtc::EmulatedNetworkNode* Build(
      std::optional<uint64_t> random_seed = std::nullopt);

 private:
  webrtc::NetworkEmulationManager& net_;
  network_behaviour::NetworkConfigSchedule schedule_;
  absl::AnyInvocable<bool(webrtc::Timestamp)> start_condition_;
};

}  // namespace webrtc

#endif  // API_TEST_NETWORK_EMULATION_SCHEDULABLE_NETWORK_NODE_BUILDER_H_
