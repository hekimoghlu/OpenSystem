/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#include "api/test/network_emulation/schedulable_network_node_builder.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "api/test/network_emulation/network_config_schedule.pb.h"
#include "api/test/network_emulation_manager.h"
#include "api/units/timestamp.h"
#include "rtc_base/time_utils.h"
#include "test/network/schedulable_network_behavior.h"

namespace webrtc {

SchedulableNetworkNodeBuilder::SchedulableNetworkNodeBuilder(
    webrtc::NetworkEmulationManager& net,
    network_behaviour::NetworkConfigSchedule schedule)
    : net_(net),
      schedule_(std::move(schedule)),
      start_condition_([](webrtc::Timestamp) { return true; }) {}

void SchedulableNetworkNodeBuilder::set_start_condition(
    absl::AnyInvocable<bool(webrtc::Timestamp)> start_condition) {
  start_condition_ = std::move(start_condition);
}

webrtc::EmulatedNetworkNode* SchedulableNetworkNodeBuilder::Build(
    std::optional<uint64_t> random_seed) {
  uint64_t seed = random_seed.has_value()
                      ? *random_seed
                      : static_cast<uint64_t>(rtc::TimeNanos());
  return net_.CreateEmulatedNode(std::make_unique<SchedulableNetworkBehavior>(
      std::move(schedule_), seed, *net_.time_controller()->GetClock(),
      std::move(start_condition_)));
}
}  // namespace webrtc
