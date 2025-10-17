/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "api/rtp_packet_info.h"

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "api/rtp_headers.h"
#include "api/units/timestamp.h"

namespace webrtc {

RtpPacketInfo::RtpPacketInfo()
    : ssrc_(0), rtp_timestamp_(0), receive_time_(Timestamp::MinusInfinity()) {}

RtpPacketInfo::RtpPacketInfo(uint32_t ssrc,
                             std::vector<uint32_t> csrcs,
                             uint32_t rtp_timestamp,
                             Timestamp receive_time)
    : ssrc_(ssrc),
      csrcs_(std::move(csrcs)),
      rtp_timestamp_(rtp_timestamp),
      receive_time_(receive_time) {}

RtpPacketInfo::RtpPacketInfo(const RTPHeader& rtp_header,
                             Timestamp receive_time)
    : ssrc_(rtp_header.ssrc),
      rtp_timestamp_(rtp_header.timestamp),
      receive_time_(receive_time) {
  const auto& extension = rtp_header.extension;
  const auto csrcs_count = std::min<size_t>(rtp_header.numCSRCs, kRtpCsrcSize);

  csrcs_.assign(&rtp_header.arrOfCSRCs[0], &rtp_header.arrOfCSRCs[csrcs_count]);

  if (extension.audio_level()) {
    audio_level_ = extension.audio_level()->level();
  }

  absolute_capture_time_ = extension.absolute_capture_time;
}

bool operator==(const RtpPacketInfo& lhs, const RtpPacketInfo& rhs) {
  return (lhs.ssrc() == rhs.ssrc()) && (lhs.csrcs() == rhs.csrcs()) &&
         (lhs.rtp_timestamp() == rhs.rtp_timestamp()) &&
         (lhs.receive_time() == rhs.receive_time()) &&
         (lhs.audio_level() == rhs.audio_level()) &&
         (lhs.absolute_capture_time() == rhs.absolute_capture_time()) &&
         (lhs.local_capture_clock_offset() == rhs.local_capture_clock_offset());
}

}  // namespace webrtc
