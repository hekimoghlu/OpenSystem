/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_FLEXFEC_HEADER_READER_WRITER_H_
#define MODULES_RTP_RTCP_SOURCE_FLEXFEC_HEADER_READER_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include "modules/rtp_rtcp/source/forward_error_correction.h"

namespace webrtc {

// FlexFEC header in flexible mode (R=0, F=0), minimum 12 bytes.
// https://datatracker.ietf.org/doc/html/rfc8627#section-4.2.2.1
//
//     0                   1                   2                   3
//     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  0 |0|0|P|X|  CC   |M| PT recovery |        length recovery        |
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  4 |                          TS recovery                          |
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  8 |           SN base_i           |k|          Mask [0-14]        |
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// 12 |k|                   Mask [15-45] (optional)                   |
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// 16 |                     Mask [46-109] (optional)                  |
// 20 |                                                               |
//    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//    |   ... next SN base and Mask for CSRC_i in CSRC list ...       |
//

class FlexfecHeaderReader : public FecHeaderReader {
 public:
  FlexfecHeaderReader();
  ~FlexfecHeaderReader() override;

  bool ReadFecHeader(
      ForwardErrorCorrection::ReceivedFecPacket* fec_packet) const override;
};

class FlexfecHeaderWriter : public FecHeaderWriter {
 public:
  FlexfecHeaderWriter();
  ~FlexfecHeaderWriter() override;

  size_t MinPacketMaskSize(const uint8_t* packet_mask,
                           size_t packet_mask_size) const override;

  size_t FecHeaderSize(size_t packet_mask_row_size) const override;

  void FinalizeFecHeader(
      rtc::ArrayView<const ProtectedStream> protected_streams,
      ForwardErrorCorrection::Packet& fec_packet) const override;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_FLEXFEC_HEADER_READER_WRITER_H_
