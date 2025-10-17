/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef API_RTP_PACKET_INFO_H_
#define API_RTP_PACKET_INFO_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "api/rtp_headers.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

//
// Structure to hold information about a received `RtpPacket`. It is primarily
// used to carry per-packet information from when a packet is received until
// the information is passed to `SourceTracker`.
//
class RTC_EXPORT RtpPacketInfo {
 public:
  RtpPacketInfo();

  RtpPacketInfo(uint32_t ssrc,
                std::vector<uint32_t> csrcs,
                uint32_t rtp_timestamp,
                Timestamp receive_time);

  RtpPacketInfo(const RTPHeader& rtp_header, Timestamp receive_time);

  RtpPacketInfo(const RtpPacketInfo& other) = default;
  RtpPacketInfo(RtpPacketInfo&& other) = default;
  RtpPacketInfo& operator=(const RtpPacketInfo& other) = default;
  RtpPacketInfo& operator=(RtpPacketInfo&& other) = default;

  uint32_t ssrc() const { return ssrc_; }
  void set_ssrc(uint32_t value) { ssrc_ = value; }

  const std::vector<uint32_t>& csrcs() const { return csrcs_; }
  void set_csrcs(std::vector<uint32_t> value) { csrcs_ = std::move(value); }

  uint32_t rtp_timestamp() const { return rtp_timestamp_; }
  void set_rtp_timestamp(uint32_t value) { rtp_timestamp_ = value; }

  Timestamp receive_time() const { return receive_time_; }
  void set_receive_time(Timestamp value) { receive_time_ = value; }

  std::optional<uint8_t> audio_level() const { return audio_level_; }
  RtpPacketInfo& set_audio_level(std::optional<uint8_t> value) {
    audio_level_ = value;
    return *this;
  }

  const std::optional<AbsoluteCaptureTime>& absolute_capture_time() const {
    return absolute_capture_time_;
  }
  RtpPacketInfo& set_absolute_capture_time(
      const std::optional<AbsoluteCaptureTime>& value) {
    absolute_capture_time_ = value;
    return *this;
  }

  const std::optional<TimeDelta>& local_capture_clock_offset() const {
    return local_capture_clock_offset_;
  }
  RtpPacketInfo& set_local_capture_clock_offset(
      std::optional<TimeDelta> value) {
    local_capture_clock_offset_ = value;
    return *this;
  }

 private:
  // Fields from the RTP header:
  // https://tools.ietf.org/html/rfc3550#section-5.1
  uint32_t ssrc_;
  std::vector<uint32_t> csrcs_;
  uint32_t rtp_timestamp_;

  // Local `webrtc::Clock`-based timestamp of when the packet was received.
  Timestamp receive_time_;

  // Fields from the Audio Level header extension:
  // https://tools.ietf.org/html/rfc6464#section-3
  std::optional<uint8_t> audio_level_;

  // Fields from the Absolute Capture Time header extension:
  // http://www.webrtc.org/experiments/rtp-hdrext/abs-capture-time
  std::optional<AbsoluteCaptureTime> absolute_capture_time_;

  // Clock offset between the local clock and the capturer's clock.
  // Do not confuse with `AbsoluteCaptureTime::estimated_capture_clock_offset`
  // which instead represents the clock offset between a remote sender and the
  // capturer. The following holds:
  //   Capture's NTP Clock = Local NTP Clock + Local-Capture Clock Offset
  std::optional<TimeDelta> local_capture_clock_offset_;
};

bool operator==(const RtpPacketInfo& lhs, const RtpPacketInfo& rhs);

inline bool operator!=(const RtpPacketInfo& lhs, const RtpPacketInfo& rhs) {
  return !(lhs == rhs);
}

}  // namespace webrtc

#endif  // API_RTP_PACKET_INFO_H_
