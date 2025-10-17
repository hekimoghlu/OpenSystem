/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "api/transport/network_types.h"

#include <algorithm>
#include <vector>

namespace webrtc {
StreamsConfig::StreamsConfig() = default;
StreamsConfig::StreamsConfig(const StreamsConfig&) = default;
StreamsConfig::~StreamsConfig() = default;

TargetRateConstraints::TargetRateConstraints() = default;
TargetRateConstraints::TargetRateConstraints(const TargetRateConstraints&) =
    default;
TargetRateConstraints::~TargetRateConstraints() = default;

NetworkRouteChange::NetworkRouteChange() = default;
NetworkRouteChange::NetworkRouteChange(const NetworkRouteChange&) = default;
NetworkRouteChange::~NetworkRouteChange() = default;

PacketResult::PacketResult() = default;
PacketResult::PacketResult(const PacketResult& other) = default;
PacketResult::~PacketResult() = default;

bool PacketResult::ReceiveTimeOrder::operator()(const PacketResult& lhs,
                                                const PacketResult& rhs) {
  if (lhs.receive_time != rhs.receive_time)
    return lhs.receive_time < rhs.receive_time;
  if (lhs.sent_packet.send_time != rhs.sent_packet.send_time)
    return lhs.sent_packet.send_time < rhs.sent_packet.send_time;
  return lhs.sent_packet.sequence_number < rhs.sent_packet.sequence_number;
}

TransportPacketsFeedback::TransportPacketsFeedback() = default;
TransportPacketsFeedback::TransportPacketsFeedback(
    const TransportPacketsFeedback& other) = default;
TransportPacketsFeedback::~TransportPacketsFeedback() = default;

std::vector<PacketResult> TransportPacketsFeedback::ReceivedWithSendInfo()
    const {
  std::vector<PacketResult> res;
  for (const PacketResult& fb : packet_feedbacks) {
    if (fb.IsReceived()) {
      res.push_back(fb);
    }
  }
  return res;
}

std::vector<PacketResult> TransportPacketsFeedback::LostWithSendInfo() const {
  std::vector<PacketResult> res;
  for (const PacketResult& fb : packet_feedbacks) {
    if (!fb.IsReceived()) {
      res.push_back(fb);
    }
  }
  return res;
}

std::vector<PacketResult> TransportPacketsFeedback::PacketsWithFeedback()
    const {
  return packet_feedbacks;
}

std::vector<PacketResult> TransportPacketsFeedback::SortedByReceiveTime()
    const {
  std::vector<PacketResult> res;
  for (const PacketResult& fb : packet_feedbacks) {
    if (fb.IsReceived()) {
      res.push_back(fb);
    }
  }
  std::sort(res.begin(), res.end(), PacketResult::ReceiveTimeOrder());
  return res;
}

NetworkControlUpdate::NetworkControlUpdate() = default;
NetworkControlUpdate::NetworkControlUpdate(const NetworkControlUpdate&) =
    default;
NetworkControlUpdate::~NetworkControlUpdate() = default;

PacedPacketInfo::PacedPacketInfo() = default;

PacedPacketInfo::PacedPacketInfo(int probe_cluster_id,
                                 int probe_cluster_min_probes,
                                 int probe_cluster_min_bytes)
    : probe_cluster_id(probe_cluster_id),
      probe_cluster_min_probes(probe_cluster_min_probes),
      probe_cluster_min_bytes(probe_cluster_min_bytes) {}

bool PacedPacketInfo::operator==(const PacedPacketInfo& rhs) const {
  return send_bitrate == rhs.send_bitrate &&
         probe_cluster_id == rhs.probe_cluster_id &&
         probe_cluster_min_probes == rhs.probe_cluster_min_probes &&
         probe_cluster_min_bytes == rhs.probe_cluster_min_bytes;
}

}  // namespace webrtc
