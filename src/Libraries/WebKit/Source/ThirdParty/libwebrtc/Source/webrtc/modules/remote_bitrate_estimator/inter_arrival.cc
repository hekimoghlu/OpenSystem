/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#include "modules/remote_bitrate_estimator/inter_arrival.h"

#include "modules/include/module_common_types_public.h"
#include "rtc_base/logging.h"

namespace webrtc {

static const int kBurstDeltaThresholdMs = 5;
static const int kMaxBurstDurationMs = 100;

InterArrival::InterArrival(uint32_t timestamp_group_length_ticks,
                           double timestamp_to_ms_coeff)
    : kTimestampGroupLengthTicks(timestamp_group_length_ticks),
      current_timestamp_group_(),
      prev_timestamp_group_(),
      timestamp_to_ms_coeff_(timestamp_to_ms_coeff),
      num_consecutive_reordered_packets_(0) {}

bool InterArrival::ComputeDeltas(uint32_t timestamp,
                                 int64_t arrival_time_ms,
                                 int64_t system_time_ms,
                                 size_t packet_size,
                                 uint32_t* timestamp_delta,
                                 int64_t* arrival_time_delta_ms,
                                 int* packet_size_delta) {
  RTC_DCHECK(timestamp_delta);
  RTC_DCHECK(arrival_time_delta_ms);
  RTC_DCHECK(packet_size_delta);
  bool calculated_deltas = false;
  if (current_timestamp_group_.IsFirstPacket()) {
    // We don't have enough data to update the filter, so we store it until we
    // have two frames of data to process.
    current_timestamp_group_.timestamp = timestamp;
    current_timestamp_group_.first_timestamp = timestamp;
    current_timestamp_group_.first_arrival_ms = arrival_time_ms;
  } else if (!PacketInOrder(timestamp)) {
    return false;
  } else if (NewTimestampGroup(arrival_time_ms, timestamp)) {
    // First packet of a later frame, the previous frame sample is ready.
    if (prev_timestamp_group_.complete_time_ms >= 0) {
      *timestamp_delta =
          current_timestamp_group_.timestamp - prev_timestamp_group_.timestamp;
      *arrival_time_delta_ms = current_timestamp_group_.complete_time_ms -
                               prev_timestamp_group_.complete_time_ms;
      // Check system time differences to see if we have an unproportional jump
      // in arrival time. In that case reset the inter-arrival computations.
      int64_t system_time_delta_ms =
          current_timestamp_group_.last_system_time_ms -
          prev_timestamp_group_.last_system_time_ms;
      if (*arrival_time_delta_ms - system_time_delta_ms >=
          kArrivalTimeOffsetThresholdMs) {
        RTC_LOG(LS_WARNING)
            << "The arrival time clock offset has changed (diff = "
            << *arrival_time_delta_ms - system_time_delta_ms
            << " ms), resetting.";
        Reset();
        return false;
      }
      if (*arrival_time_delta_ms < 0) {
        // The group of packets has been reordered since receiving its local
        // arrival timestamp.
        ++num_consecutive_reordered_packets_;
        if (num_consecutive_reordered_packets_ >= kReorderedResetThreshold) {
          RTC_LOG(LS_WARNING)
              << "Packets are being reordered on the path from the "
                 "socket to the bandwidth estimator. Ignoring this "
                 "packet for bandwidth estimation, resetting.";
          Reset();
        }
        return false;
      } else {
        num_consecutive_reordered_packets_ = 0;
      }
      RTC_DCHECK_GE(*arrival_time_delta_ms, 0);
      *packet_size_delta = static_cast<int>(current_timestamp_group_.size) -
                           static_cast<int>(prev_timestamp_group_.size);
      calculated_deltas = true;
    }
    prev_timestamp_group_ = current_timestamp_group_;
    // The new timestamp is now the current frame.
    current_timestamp_group_.first_timestamp = timestamp;
    current_timestamp_group_.timestamp = timestamp;
    current_timestamp_group_.first_arrival_ms = arrival_time_ms;
    current_timestamp_group_.size = 0;
  } else {
    current_timestamp_group_.timestamp =
        LatestTimestamp(current_timestamp_group_.timestamp, timestamp);
  }
  // Accumulate the frame size.
  current_timestamp_group_.size += packet_size;
  current_timestamp_group_.complete_time_ms = arrival_time_ms;
  current_timestamp_group_.last_system_time_ms = system_time_ms;

  return calculated_deltas;
}

bool InterArrival::PacketInOrder(uint32_t timestamp) {
  if (current_timestamp_group_.IsFirstPacket()) {
    return true;
  } else {
    // Assume that a diff which is bigger than half the timestamp interval
    // (32 bits) must be due to reordering. This code is almost identical to
    // that in IsNewerTimestamp() in module_common_types.h.
    uint32_t timestamp_diff =
        timestamp - current_timestamp_group_.first_timestamp;
    return timestamp_diff < 0x80000000;
  }
}

// Assumes that `timestamp` is not reordered compared to
// `current_timestamp_group_`.
bool InterArrival::NewTimestampGroup(int64_t arrival_time_ms,
                                     uint32_t timestamp) const {
  if (current_timestamp_group_.IsFirstPacket()) {
    return false;
  } else if (BelongsToBurst(arrival_time_ms, timestamp)) {
    return false;
  } else {
    uint32_t timestamp_diff =
        timestamp - current_timestamp_group_.first_timestamp;
    return timestamp_diff > kTimestampGroupLengthTicks;
  }
}

bool InterArrival::BelongsToBurst(int64_t arrival_time_ms,
                                  uint32_t timestamp) const {
  RTC_DCHECK_GE(current_timestamp_group_.complete_time_ms, 0);
  int64_t arrival_time_delta_ms =
      arrival_time_ms - current_timestamp_group_.complete_time_ms;
  uint32_t timestamp_diff = timestamp - current_timestamp_group_.timestamp;
  int64_t ts_delta_ms = timestamp_to_ms_coeff_ * timestamp_diff + 0.5;
  if (ts_delta_ms == 0)
    return true;
  int propagation_delta_ms = arrival_time_delta_ms - ts_delta_ms;
  if (propagation_delta_ms < 0 &&
      arrival_time_delta_ms <= kBurstDeltaThresholdMs &&
      arrival_time_ms - current_timestamp_group_.first_arrival_ms <
          kMaxBurstDurationMs)
    return true;
  return false;
}

void InterArrival::Reset() {
  num_consecutive_reordered_packets_ = 0;
  current_timestamp_group_ = TimestampGroup();
  prev_timestamp_group_ = TimestampGroup();
}
}  // namespace webrtc
