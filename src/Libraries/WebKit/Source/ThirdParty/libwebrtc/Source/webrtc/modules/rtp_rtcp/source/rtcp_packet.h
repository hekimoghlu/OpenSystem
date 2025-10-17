/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_H_

#include <stddef.h>
#include <stdint.h>

#include "api/array_view.h"
#include "api/function_view.h"
#include "rtc_base/buffer.h"

namespace webrtc {
namespace rtcp {
// Class for building RTCP packets.
//
//  Example:
//  ReportBlock report_block;
//  report_block.SetMediaSsrc(234);
//  report_block.SetFractionLost(10);
//
//  ReceiverReport rr;
//  rr.SetSenderSsrc(123);
//  rr.AddReportBlock(report_block);
//
//  Fir fir;
//  fir.SetSenderSsrc(123);
//  fir.AddRequestTo(234, 56);
//
//  size_t length = 0;                     // Builds an intra frame request
//  uint8_t packet[kPacketSize];           // with sequence number 56.
//  fir.Build(packet, &length, kPacketSize);
//
//  rtc::Buffer packet = fir.Build();      // Returns a RawPacket holding
//                                         // the built rtcp packet.
//
//  CompoundPacket compound;               // Builds a compound RTCP packet with
//  compound.Append(&rr);                  // a receiver report, report block
//  compound.Append(&fir);                 // and fir message.
//  rtc::Buffer packet = compound.Build();

class RtcpPacket {
 public:
  // Callback used to signal that an RTCP packet is ready. Note that this may
  // not contain all data in this RtcpPacket; if a packet cannot fit in
  // max_length bytes, it will be fragmented and multiple calls to this
  // callback will be made.
  using PacketReadyCallback =
      rtc::FunctionView<void(rtc::ArrayView<const uint8_t> packet)>;

  virtual ~RtcpPacket() = default;

  void SetSenderSsrc(uint32_t ssrc) { sender_ssrc_ = ssrc; }
  uint32_t sender_ssrc() const { return sender_ssrc_; }

  // Convenience method mostly used for test. Creates packet without
  // fragmentation using BlockLength() to allocate big enough buffer.
  rtc::Buffer Build() const;

  // Returns true if call to Create succeeded.
  bool Build(size_t max_length, PacketReadyCallback callback) const;

  // Size of this packet in bytes (including headers).
  virtual size_t BlockLength() const = 0;

  // Creates packet in the given buffer at the given position.
  // Calls PacketReadyCallback::OnPacketReady if remaining buffer is too small
  // and assume buffer can be reused after OnPacketReady returns.
  virtual bool Create(uint8_t* packet,
                      size_t* index,
                      size_t max_length,
                      PacketReadyCallback callback) const = 0;

 protected:
  // Size of the rtcp common header.
  static constexpr size_t kHeaderLength = 4;
  RtcpPacket() {}

  static void CreateHeader(size_t count_or_format,
                           uint8_t packet_type,
                           size_t block_length,  // Payload size in 32bit words.
                           uint8_t* buffer,
                           size_t* pos);

  static void CreateHeader(size_t count_or_format,
                           uint8_t packet_type,
                           size_t block_length,  // Payload size in 32bit words.
                           bool padding,  // True if there are padding bytes.
                           uint8_t* buffer,
                           size_t* pos);

  bool OnBufferFull(uint8_t* packet,
                    size_t* index,
                    PacketReadyCallback callback) const;
  // Size of the rtcp packet as written in header.
  size_t HeaderLength() const;

 private:
  uint32_t sender_ssrc_ = 0;
};
}  // namespace rtcp
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_H_
