/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTP_PACKET_OUTGOING_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTP_PACKET_OUTGOING_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/events/logged_rtp_rtcp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"
#include "modules/rtp_rtcp/source/rtp_packet.h"

namespace webrtc {

class RtpPacketToSend;

class RtcEventRtpPacketOutgoing final : public RtcEvent {
 public:
  static constexpr Type kType = Type::RtpPacketOutgoing;

  RtcEventRtpPacketOutgoing(const RtpPacketToSend& packet,
                            int probe_cluster_id);
  ~RtcEventRtpPacketOutgoing() override;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventRtpPacketOutgoing> Copy() const;

  size_t packet_length() const { return packet_.size(); }

  rtc::ArrayView<const uint8_t> RawHeader() const {
    return rtc::MakeArrayView(packet_.data(), header_length());
  }
  uint32_t Ssrc() const { return packet_.Ssrc(); }
  uint32_t Timestamp() const { return packet_.Timestamp(); }
  uint16_t SequenceNumber() const { return packet_.SequenceNumber(); }
  uint8_t PayloadType() const { return packet_.PayloadType(); }
  bool Marker() const { return packet_.Marker(); }
  template <typename ExtensionTrait, typename... Args>
  bool GetExtension(Args&&... args) const {
    return packet_.GetExtension<ExtensionTrait>(std::forward<Args>(args)...);
  }
  template <typename ExtensionTrait>
  rtc::ArrayView<const uint8_t> GetRawExtension() const {
    return packet_.GetRawExtension<ExtensionTrait>();
  }
  template <typename ExtensionTrait>
  bool HasExtension() const {
    return packet_.HasExtension<ExtensionTrait>();
  }

  size_t payload_length() const { return packet_.payload_size(); }
  size_t header_length() const { return packet_.headers_size(); }
  size_t padding_length() const { return packet_.padding_size(); }
  int probe_cluster_id() const { return probe_cluster_id_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::map<uint32_t, std::vector<LoggedRtpPacketOutgoing>>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventRtpPacketOutgoing(const RtcEventRtpPacketOutgoing& other);

  const RtpPacket packet_;
  // TODO(eladalon): Delete `probe_cluster_id_` along with legacy encoding.
  const int probe_cluster_id_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTP_PACKET_OUTGOING_H_
