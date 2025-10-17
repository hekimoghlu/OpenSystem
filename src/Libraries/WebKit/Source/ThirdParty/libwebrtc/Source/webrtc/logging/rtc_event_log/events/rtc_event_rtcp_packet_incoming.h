/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTCP_PACKET_INCOMING_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTCP_PACKET_INCOMING_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/events/logged_rtp_rtcp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class RtcEventRtcpPacketIncoming final : public RtcEvent {
 public:
  static constexpr Type kType = Type::RtcpPacketIncoming;

  explicit RtcEventRtcpPacketIncoming(rtc::ArrayView<const uint8_t> packet);
  ~RtcEventRtcpPacketIncoming() override;

  Type GetType() const override { return kType; }
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventRtcpPacketIncoming> Copy() const;

  const rtc::Buffer& packet() const { return packet_; }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedRtcpPacketIncoming>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventRtcpPacketIncoming(const RtcEventRtcpPacketIncoming& other);

  rtc::Buffer packet_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_RTCP_PACKET_INCOMING_H_
