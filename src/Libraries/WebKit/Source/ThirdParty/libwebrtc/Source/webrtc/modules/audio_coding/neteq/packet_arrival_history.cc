/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#include "modules/audio_coding/neteq/packet_arrival_history.h"

#include <algorithm>
#include <cstdint>

#include "api/neteq/tick_timer.h"
#include "rtc_base/checks.h"

namespace webrtc {

PacketArrivalHistory::PacketArrivalHistory(const TickTimer* tick_timer,
                                           int window_size_ms)
    : tick_timer_(tick_timer), window_size_ms_(window_size_ms) {}

bool PacketArrivalHistory::Insert(uint32_t rtp_timestamp,
                                  int packet_length_samples) {
  int64_t arrival_timestamp =
      tick_timer_->ticks() * tick_timer_->ms_per_tick() * sample_rate_khz_;
  PacketArrival packet(timestamp_unwrapper_.Unwrap(rtp_timestamp),
                       arrival_timestamp, packet_length_samples);
  if (IsObsolete(packet)) {
    return false;
  }
  if (Contains(packet)) {
    return false;
  }
  history_.emplace(packet.rtp_timestamp, packet);
  if (packet != history_.rbegin()->second) {
    // Packet was reordered.
    return true;
  }
  // Remove old packets.
  while (IsObsolete(history_.begin()->second)) {
    if (history_.begin()->second == min_packet_arrivals_.front()) {
      min_packet_arrivals_.pop_front();
    }
    if (history_.begin()->second == max_packet_arrivals_.front()) {
      max_packet_arrivals_.pop_front();
    }
    history_.erase(history_.begin());
  }
  // Ensure ordering constraints.
  while (!min_packet_arrivals_.empty() &&
         packet <= min_packet_arrivals_.back()) {
    min_packet_arrivals_.pop_back();
  }
  while (!max_packet_arrivals_.empty() &&
         packet >= max_packet_arrivals_.back()) {
    max_packet_arrivals_.pop_back();
  }
  min_packet_arrivals_.push_back(packet);
  max_packet_arrivals_.push_back(packet);
  return true;
}

void PacketArrivalHistory::Reset() {
  history_.clear();
  min_packet_arrivals_.clear();
  max_packet_arrivals_.clear();
  timestamp_unwrapper_.Reset();
}

int PacketArrivalHistory::GetDelayMs(uint32_t rtp_timestamp) const {
  int64_t unwrapped_rtp_timestamp =
      timestamp_unwrapper_.PeekUnwrap(rtp_timestamp);
  int64_t current_timestamp =
      tick_timer_->ticks() * tick_timer_->ms_per_tick() * sample_rate_khz_;
  PacketArrival packet(unwrapped_rtp_timestamp, current_timestamp,
                       /*duration_ms=*/0);
  return GetPacketArrivalDelayMs(packet);
}

int PacketArrivalHistory::GetMaxDelayMs() const {
  if (max_packet_arrivals_.empty()) {
    return 0;
  }
  return GetPacketArrivalDelayMs(max_packet_arrivals_.front());
}

bool PacketArrivalHistory::IsNewestRtpTimestamp(uint32_t rtp_timestamp) const {
  if (history_.empty()) {
    return true;
  }
  int64_t unwrapped_rtp_timestamp =
      timestamp_unwrapper_.PeekUnwrap(rtp_timestamp);
  return unwrapped_rtp_timestamp == history_.rbegin()->second.rtp_timestamp;
}

int PacketArrivalHistory::GetPacketArrivalDelayMs(
    const PacketArrival& packet_arrival) const {
  if (min_packet_arrivals_.empty()) {
    return 0;
  }
  RTC_DCHECK_NE(sample_rate_khz_, 0);
  // TODO(jakobi): Timestamps are first converted to millis for bit-exactness.
  return std::max<int>(
      packet_arrival.arrival_timestamp / sample_rate_khz_ -
          min_packet_arrivals_.front().arrival_timestamp / sample_rate_khz_ -
          (packet_arrival.rtp_timestamp / sample_rate_khz_ -
           min_packet_arrivals_.front().rtp_timestamp / sample_rate_khz_),
      0);
}

bool PacketArrivalHistory::IsObsolete(
    const PacketArrival& packet_arrival) const {
  if (history_.empty()) {
    return false;
  }
  return packet_arrival.rtp_timestamp + window_size_ms_ * sample_rate_khz_ <
         history_.rbegin()->second.rtp_timestamp;
}

bool PacketArrivalHistory::Contains(const PacketArrival& packet_arrival) const {
  auto it = history_.upper_bound(packet_arrival.rtp_timestamp);
  if (it == history_.begin()) {
    return false;
  }
  --it;
  return it->second.contains(packet_arrival);
}

}  // namespace webrtc
