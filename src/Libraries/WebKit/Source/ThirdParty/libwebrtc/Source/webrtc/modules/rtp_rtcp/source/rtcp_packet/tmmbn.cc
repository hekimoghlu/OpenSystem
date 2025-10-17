/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_packet/tmmbn.h"

#include "modules/rtp_rtcp/source/rtcp_packet/common_header.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace rtcp {
// RFC 4585: Feedback format.
// Common packet format:
//
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |V=2|P|   FMT   |       PT      |          length               |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |                  SSRC of packet sender                        |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |             SSRC of media source (unused) = 0                 |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   :            Feedback Control Information (FCI)                 :
//   :                                                               :
// Temporary Maximum Media Stream Bit Rate Notification (TMMBN) (RFC 5104).
// The Feedback Control Information (FCI) consists of zero, one, or more
// TMMBN FCI entries.
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |                              SSRC                             |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   | MxTBR Exp |  MxTBR Mantissa                 |Measured Overhead|
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Tmmbn::Tmmbn() = default;

Tmmbn::~Tmmbn() = default;

bool Tmmbn::Parse(const CommonHeader& packet) {
  RTC_DCHECK_EQ(packet.type(), kPacketType);
  RTC_DCHECK_EQ(packet.fmt(), kFeedbackMessageType);

  if (packet.payload_size_bytes() < kCommonFeedbackLength) {
    RTC_LOG(LS_WARNING) << "Payload length " << packet.payload_size_bytes()
                        << " is too small for TMMBN.";
    return false;
  }
  size_t items_size_bytes = packet.payload_size_bytes() - kCommonFeedbackLength;
  if (items_size_bytes % TmmbItem::kLength != 0) {
    RTC_LOG(LS_WARNING) << "Payload length " << packet.payload_size_bytes()
                        << " is not valid for TMMBN.";
    return false;
  }
  ParseCommonFeedback(packet.payload());
  const uint8_t* next_item = packet.payload() + kCommonFeedbackLength;

  size_t number_of_items = items_size_bytes / TmmbItem::kLength;
  items_.resize(number_of_items);
  for (TmmbItem& item : items_) {
    if (!item.Parse(next_item))
      return false;
    next_item += TmmbItem::kLength;
  }
  return true;
}

void Tmmbn::AddTmmbr(const TmmbItem& item) {
  items_.push_back(item);
}

size_t Tmmbn::BlockLength() const {
  return kHeaderLength + kCommonFeedbackLength +
         TmmbItem::kLength * items_.size();
}

bool Tmmbn::Create(uint8_t* packet,
                   size_t* index,
                   size_t max_length,
                   PacketReadyCallback callback) const {
  while (*index + BlockLength() > max_length) {
    if (!OnBufferFull(packet, index, callback))
      return false;
  }
  const size_t index_end = *index + BlockLength();

  CreateHeader(kFeedbackMessageType, kPacketType, HeaderLength(), packet,
               index);
  RTC_DCHECK_EQ(0, Rtpfb::media_ssrc());
  CreateCommonFeedback(packet + *index);
  *index += kCommonFeedbackLength;
  for (const TmmbItem& item : items_) {
    item.Create(packet + *index);
    *index += TmmbItem::kLength;
  }
  RTC_CHECK_EQ(index_end, *index);
  return true;
}
}  // namespace rtcp
}  // namespace webrtc
