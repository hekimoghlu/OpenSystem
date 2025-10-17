/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#ifndef TEST_NETWORK_SCHEDULABLE_NETWORK_BEHAVIOR_H_
#define TEST_NETWORK_SCHEDULABLE_NETWORK_BEHAVIOR_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "api/sequence_checker.h"
#include "api/test/network_emulation/network_config_schedule.pb.h"
#include "api/test/simulated_network.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"
#include "test/network/simulated_network.h"

namespace webrtc {

// Network behaviour implementation where parameters change over time as
// specified with a schedule proto.
class SchedulableNetworkBehavior : public SimulatedNetwork {
 public:
  SchedulableNetworkBehavior(
      network_behaviour::NetworkConfigSchedule schedule,
      uint64_t random_seed,
      Clock& clock,
      absl::AnyInvocable<bool(webrtc::Timestamp)> start_condition =
          [](webrtc::Timestamp) { return true; });

  bool EnqueuePacket(PacketInFlightInfo packet_info) override;

 private:
  TimeDelta UpdateConfigAndReschedule();

  SequenceChecker sequence_checker_;
  const network_behaviour::NetworkConfigSchedule schedule_;
  absl::AnyInvocable<bool(webrtc::Timestamp)> start_condition_
      RTC_GUARDED_BY(&sequence_checker_);
  // Send time of the first packet enqueued after `start_condition_` return
  // true.
  Timestamp first_send_time_ RTC_GUARDED_BY(&sequence_checker_) =
      Timestamp::MinusInfinity();

  Clock& clock_ RTC_GUARDED_BY(&sequence_checker_);
  BuiltInNetworkBehaviorConfig config_ RTC_GUARDED_BY(&sequence_checker_);
  // Index of the next schedule item to apply.
  int next_schedule_index_ RTC_GUARDED_BY(&sequence_checker_) = 0;
  // Total time from the first sent packet, until the last time the schedule
  // repeat.
  TimeDelta wrap_time_delta_ RTC_GUARDED_BY(&sequence_checker_) =
      TimeDelta::Zero();
  RepeatingTaskHandle schedule_task_ RTC_GUARDED_BY(&sequence_checker_);
};

}  // namespace webrtc

#endif  // TEST_NETWORK_SCHEDULABLE_NETWORK_BEHAVIOR_H_
