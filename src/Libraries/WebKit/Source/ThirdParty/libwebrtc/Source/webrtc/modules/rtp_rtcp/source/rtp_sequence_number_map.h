/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_SEQUENCE_NUMBER_MAP_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_SEQUENCE_NUMBER_MAP_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace webrtc {

// Records the association of RTP sequence numbers to timestamps and to whether
// the packet was first and/or last in the frame.
//
// 1. Limits number of entries. Whenever `max_entries` is about to be exceeded,
//    the size is reduced by approximately 25%.
// 2. RTP sequence numbers wrap around relatively infrequently.
//    This class therefore only remembers at most the last 2^15 RTP packets,
//    so that the newest packet's sequence number is still AheadOf the oldest
//    packet's sequence number.
// 3. Media frames are sometimes split into several RTP packets.
//    In such a case, Insert() is expected to be called once for each packet.
//    The timestamp is not expected to change between those calls.
class RtpSequenceNumberMap final {
 public:
  struct Info final {
    Info(uint32_t timestamp, bool is_first, bool is_last)
        : timestamp(timestamp), is_first(is_first), is_last(is_last) {}

    friend bool operator==(const Info& lhs, const Info& rhs) {
      return lhs.timestamp == rhs.timestamp && lhs.is_first == rhs.is_first &&
             lhs.is_last == rhs.is_last;
    }

    uint32_t timestamp;
    bool is_first;
    bool is_last;
  };

  explicit RtpSequenceNumberMap(size_t max_entries);
  RtpSequenceNumberMap(const RtpSequenceNumberMap& other) = delete;
  RtpSequenceNumberMap& operator=(const RtpSequenceNumberMap& other) = delete;
  ~RtpSequenceNumberMap();

  void InsertPacket(uint16_t sequence_number, Info info);
  void InsertFrame(uint16_t first_sequence_number,
                   size_t packet_count,
                   uint32_t timestamp);

  std::optional<Info> Get(uint16_t sequence_number) const;

  size_t AssociationCountForTesting() const;

 private:
  struct Association {
    explicit Association(uint16_t sequence_number)
        : Association(sequence_number, Info(0, false, false)) {}

    Association(uint16_t sequence_number, Info info)
        : sequence_number(sequence_number), info(info) {}

    uint16_t sequence_number;
    Info info;
  };

  const size_t max_entries_;

  // The non-transitivity of AheadOf() would be problematic with a map,
  // so we use a deque instead.
  std::deque<Association> associations_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_SEQUENCE_NUMBER_MAP_H_
