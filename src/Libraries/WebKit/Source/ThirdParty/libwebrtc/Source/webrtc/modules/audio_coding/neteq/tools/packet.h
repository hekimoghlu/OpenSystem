/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_H_

#include <list>

#include "api/array_view.h"
#include "api/rtp_headers.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "rtc_base/copy_on_write_buffer.h"

namespace webrtc {
namespace test {

// Class for handling RTP packets in test applications.
class Packet {
 public:
  // Creates a packet, with the packet payload (including header bytes) in
  // `packet`. The `time_ms` is an extra time associated with this packet,
  // typically used to denote arrival time.
  // `virtual_packet_length_bytes` is typically used when reading RTP dump files
  // that only contain the RTP headers, and no payload (a.k.a RTP dummy files or
  // RTP light). The `virtual_packet_length_bytes` tells what size the packet
  // had on wire, including the now discarded payload.
  Packet(rtc::CopyOnWriteBuffer packet,
         size_t virtual_packet_length_bytes,
         double time_ms,
         const RtpHeaderExtensionMap* extension_map = nullptr);

  Packet(rtc::CopyOnWriteBuffer packet,
         double time_ms,
         const RtpHeaderExtensionMap* extension_map = nullptr)
      : Packet(packet, packet.size(), time_ms, extension_map) {}

  // Same as above, but creates the packet from an already parsed RTPHeader.
  // This is typically used when reading RTP dump files that only contain the
  // RTP headers, and no payload. The `virtual_packet_length_bytes` tells what
  // size the packet had on wire, including the now discarded payload,
  // The `virtual_payload_length_bytes` tells the size of the payload.
  Packet(const RTPHeader& header,
         size_t virtual_packet_length_bytes,
         size_t virtual_payload_length_bytes,
         double time_ms);

  virtual ~Packet();

  Packet(const Packet&) = delete;
  Packet& operator=(const Packet&) = delete;

  // Parses the first bytes of the RTP payload, interpreting them as RED headers
  // according to RFC 2198. The headers will be inserted into `headers`. The
  // caller of the method assumes ownership of the objects in the list, and
  // must delete them properly.
  bool ExtractRedHeaders(std::list<RTPHeader*>* headers) const;

  // Deletes all RTPHeader objects in `headers`, but does not delete `headers`
  // itself.
  static void DeleteRedHeaders(std::list<RTPHeader*>* headers);

  const uint8_t* payload() const { return rtp_payload_.data(); }

  size_t packet_length_bytes() const { return packet_.size(); }

  size_t payload_length_bytes() const { return rtp_payload_.size(); }

  size_t virtual_packet_length_bytes() const {
    return virtual_packet_length_bytes_;
  }

  size_t virtual_payload_length_bytes() const {
    return virtual_payload_length_bytes_;
  }

  const RTPHeader& header() const { return header_; }

  double time_ms() const { return time_ms_; }
  bool valid_header() const { return valid_header_; }

 private:
  bool ParseHeader(const RtpHeaderExtensionMap* extension_map);
  void CopyToHeader(RTPHeader* destination) const;

  RTPHeader header_;
  const rtc::CopyOnWriteBuffer packet_;
  rtc::ArrayView<const uint8_t> rtp_payload_;  // Empty for dummy RTP packets.
  // Virtual lengths are used when parsing RTP header files (dummy RTP files).
  const size_t virtual_packet_length_bytes_;
  size_t virtual_payload_length_bytes_ = 0;
  const double time_ms_;  // Used to denote a packet's arrival time.
  const bool valid_header_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_PACKET_H_
