/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#include "modules/congestion_controller/pcc/monitor_interval.h"

#include <stddef.h>

#include <cmath>

#include "rtc_base/logging.h"

namespace webrtc {
namespace pcc {

PccMonitorInterval::PccMonitorInterval(DataRate target_sending_rate,
                                       Timestamp start_time,
                                       TimeDelta duration)
    : target_sending_rate_(target_sending_rate),
      start_time_(start_time),
      interval_duration_(duration),
      received_packets_size_(DataSize::Zero()),
      feedback_collection_done_(false) {}

PccMonitorInterval::~PccMonitorInterval() = default;

PccMonitorInterval::PccMonitorInterval(const PccMonitorInterval& other) =
    default;

void PccMonitorInterval::OnPacketsFeedback(
    const std::vector<PacketResult>& packets_results) {
  for (const PacketResult& packet_result : packets_results) {
    if (packet_result.sent_packet.send_time <= start_time_) {
      continue;
    }
    // Here we assume that if some packets are reordered with packets sent
    // after the end of the monitor interval, then they are lost. (Otherwise
    // it is not clear how long should we wait for packets feedback to arrive).
    if (packet_result.sent_packet.send_time >
        start_time_ + interval_duration_) {
      feedback_collection_done_ = true;
      return;
    }
    if (!packet_result.IsReceived()) {
      lost_packets_sent_time_.push_back(packet_result.sent_packet.send_time);
    } else {
      received_packets_.push_back(
          {packet_result.receive_time - packet_result.sent_packet.send_time,
           packet_result.sent_packet.send_time});
      received_packets_size_ += packet_result.sent_packet.size;
    }
  }
}

// For the formula used in computations see formula for "slope" in the second
// method:
// https://www.johndcook.com/blog/2008/10/20/comparing-two-ways-to-fit-a-line-to-data/
double PccMonitorInterval::ComputeDelayGradient(
    double delay_gradient_threshold) const {
  // Early return to prevent division by 0 in case all packets are sent at the
  // same time.
  if (received_packets_.empty() || received_packets_.front().sent_time ==
                                       received_packets_.back().sent_time) {
    return 0;
  }
  double sum_times = 0;
  for (const ReceivedPacket& packet : received_packets_) {
    double time_delta_us =
        (packet.sent_time - received_packets_[0].sent_time).us();
    sum_times += time_delta_us;
  }
  double sum_squared_scaled_time_deltas = 0;
  double sum_scaled_time_delta_dot_delay = 0;
  for (const ReceivedPacket& packet : received_packets_) {
    double time_delta_us =
        (packet.sent_time - received_packets_[0].sent_time).us();
    double delay = packet.delay.us();
    double scaled_time_delta_us =
        time_delta_us - sum_times / received_packets_.size();
    sum_squared_scaled_time_deltas +=
        scaled_time_delta_us * scaled_time_delta_us;
    sum_scaled_time_delta_dot_delay += scaled_time_delta_us * delay;
  }
  double rtt_gradient =
      sum_scaled_time_delta_dot_delay / sum_squared_scaled_time_deltas;
  if (std::abs(rtt_gradient) < delay_gradient_threshold)
    rtt_gradient = 0;
  return rtt_gradient;
}

bool PccMonitorInterval::IsFeedbackCollectionDone() const {
  return feedback_collection_done_;
}

Timestamp PccMonitorInterval::GetEndTime() const {
  return start_time_ + interval_duration_;
}

double PccMonitorInterval::GetLossRate() const {
  size_t packets_lost = lost_packets_sent_time_.size();
  size_t packets_received = received_packets_.size();
  if (packets_lost == 0)
    return 0;
  return static_cast<double>(packets_lost) / (packets_lost + packets_received);
}

DataRate PccMonitorInterval::GetTargetSendingRate() const {
  return target_sending_rate_;
}

DataRate PccMonitorInterval::GetTransmittedPacketsRate() const {
  if (received_packets_.empty()) {
    return target_sending_rate_;
  }
  Timestamp receive_time_of_first_packet =
      received_packets_.front().sent_time + received_packets_.front().delay;
  Timestamp receive_time_of_last_packet =
      received_packets_.back().sent_time + received_packets_.back().delay;
  if (receive_time_of_first_packet == receive_time_of_last_packet) {
    RTC_LOG(LS_WARNING)
        << "All packets in monitor interval were received at the same time.";
    return target_sending_rate_;
  }
  return received_packets_size_ /
         (receive_time_of_last_packet - receive_time_of_first_packet);
}

}  // namespace pcc
}  // namespace webrtc
