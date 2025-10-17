/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include "test/network/schedulable_network_behavior.h"

#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "api/sequence_checker.h"
#include "api/test/network_emulation/network_config_schedule.pb.h"
#include "api/test/simulated_network.h"
#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "system_wrappers/include/clock.h"
#include "test/network/simulated_network.h"

namespace webrtc {

namespace {

using ::webrtc::BuiltInNetworkBehaviorConfig;

void UpdateConfigFromSchedule(
    const network_behaviour::NetworkConfigScheduleItem& schedule_item,
    BuiltInNetworkBehaviorConfig& config) {
  if (schedule_item.has_queue_length_packets()) {
    config.queue_length_packets = schedule_item.queue_length_packets();
  }
  if (schedule_item.has_queue_delay_ms()) {
    config.queue_delay_ms = schedule_item.queue_delay_ms();
  }
  if (schedule_item.has_link_capacity_kbps()) {
    config.link_capacity =
        DataRate::KilobitsPerSec(schedule_item.link_capacity_kbps());
  }
  if (schedule_item.has_loss_percent()) {
    config.loss_percent = schedule_item.loss_percent();
  }
  if (schedule_item.has_delay_standard_deviation_ms()) {
    config.delay_standard_deviation_ms =
        schedule_item.delay_standard_deviation_ms();
  }
  if (schedule_item.has_allow_reordering()) {
    config.allow_reordering = schedule_item.allow_reordering();
  }
  if (schedule_item.has_avg_burst_loss_length()) {
    config.avg_burst_loss_length = schedule_item.avg_burst_loss_length();
  }
  if (schedule_item.has_packet_overhead()) {
    config.packet_overhead = schedule_item.packet_overhead();
  }
}

BuiltInNetworkBehaviorConfig GetInitialConfig(
    const network_behaviour::NetworkConfigSchedule& schedule) {
  BuiltInNetworkBehaviorConfig config;
  if (!schedule.item().empty()) {
    UpdateConfigFromSchedule(schedule.item(0), config);
  }
  return config;
}

}  // namespace

SchedulableNetworkBehavior::SchedulableNetworkBehavior(
    network_behaviour::NetworkConfigSchedule schedule,
    uint64_t random_seed,
    webrtc::Clock& clock,
    absl::AnyInvocable<bool(webrtc::Timestamp)> start_callback)
    : SimulatedNetwork(GetInitialConfig(schedule), random_seed),
      schedule_(std::move(schedule)),
      start_condition_(std::move(start_callback)),
      clock_(clock),
      config_(GetInitialConfig(schedule_)) {
  if (schedule_.item().size() > 1) {
    next_schedule_index_ = 1;
  }
  sequence_checker_.Detach();
}

bool SchedulableNetworkBehavior::EnqueuePacket(
    webrtc::PacketInFlightInfo packet_info) {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  if (first_send_time_.IsInfinite() &&
      start_condition_(webrtc::Timestamp::Micros(packet_info.send_time_us))) {
    first_send_time_ = webrtc::Timestamp::Micros(packet_info.send_time_us);
    if (schedule_.item().size() > 1) {
      RTC_CHECK_LT(next_schedule_index_, schedule_.item().size());
      webrtc::TimeDelta delay =
          webrtc::TimeDelta::Millis(schedule_.item()[next_schedule_index_]
                                        .time_since_first_sent_packet_ms());
      schedule_task_ = RepeatingTaskHandle::DelayedStart(
          webrtc::TaskQueueBase::Current(), delay,
          [this] { return UpdateConfigAndReschedule(); });
    }
  }
  return SimulatedNetwork::EnqueuePacket(packet_info);
}

TimeDelta SchedulableNetworkBehavior::UpdateConfigAndReschedule() {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  Timestamp reschedule_time = clock_.CurrentTime();
  RTC_CHECK_LT(next_schedule_index_, schedule_.item().size());

  auto next_config = schedule_.item()[next_schedule_index_];
  UpdateConfigFromSchedule(next_config, config_);
  SimulatedNetwork::SetConfig(config_, reschedule_time);
  next_schedule_index_ = ++next_schedule_index_ % schedule_.item().size();
  webrtc::TimeDelta delay = webrtc::TimeDelta::Zero();
  webrtc::TimeDelta time_since_first_sent_packet =
      reschedule_time - first_send_time_;
  if (next_schedule_index_ != 0) {
    delay = std::max(TimeDelta::Millis(schedule_.item()[next_schedule_index_]
                                           .time_since_first_sent_packet_ms()) -
                         (time_since_first_sent_packet - wrap_time_delta_),
                     TimeDelta::Zero());
  } else if (!schedule_.has_repeat_schedule_after_last_ms()) {
    // No more schedule items.
    schedule_task_.Stop();
    return TimeDelta::Zero();  // This is ignored.
  } else {
    // Wrap around to the first schedule item.
    wrap_time_delta_ +=
        TimeDelta::Millis(schedule_.repeat_schedule_after_last_ms()) +
        TimeDelta::Millis(schedule_.item()[schedule_.item().size() - 1]
                              .time_since_first_sent_packet_ms());
    delay =
        webrtc::TimeDelta::Millis(schedule_.repeat_schedule_after_last_ms());
    RTC_DCHECK_GE(delay, TimeDelta::Zero());
  }

  return delay;
}

}  // namespace webrtc
