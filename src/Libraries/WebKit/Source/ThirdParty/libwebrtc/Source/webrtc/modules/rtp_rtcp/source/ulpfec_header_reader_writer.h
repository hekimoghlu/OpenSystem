/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_ULPFEC_HEADER_READER_WRITER_H_
#define MODULES_RTP_RTCP_SOURCE_ULPFEC_HEADER_READER_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/rtp_rtcp/source/forward_error_correction.h"

namespace webrtc {

// FEC Level 0 Header, 10 bytes.
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |E|L|P|X|  CC   |M| PT recovery |            SN base            |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |                          TS recovery                          |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |        length recovery        |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// FEC Level 1 Header, 4 bytes (L = 0) or 8 bytes (L = 1).
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |       Protection Length       |             mask              |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |              mask cont. (present only when L = 1)             |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class UlpfecHeaderReader : public FecHeaderReader {
 public:
  UlpfecHeaderReader();
  ~UlpfecHeaderReader() override;

  bool ReadFecHeader(
      ForwardErrorCorrection::ReceivedFecPacket* fec_packet) const override;
};

class UlpfecHeaderWriter : public FecHeaderWriter {
 public:
  UlpfecHeaderWriter();
  ~UlpfecHeaderWriter() override;

  size_t MinPacketMaskSize(const uint8_t* packet_mask,
                           size_t packet_mask_size) const override;

  size_t FecHeaderSize(size_t packet_mask_row_size) const override;

  void FinalizeFecHeader(
      rtc::ArrayView<const ProtectedStream> protected_streams,
      ForwardErrorCorrection::Packet& fec_packet) const override;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_ULPFEC_HEADER_READER_WRITER_H_
