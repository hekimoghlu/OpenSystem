/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include <stddef.h>
#include <stdint.h>

#include "api/video/video_frame_type.h"
#include "modules/rtp_rtcp/source/rtp_format.h"
#include "modules/rtp_rtcp/source/rtp_format_vp8.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"
#if WEBRTC_WEBKIT_BUILD
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_vp8.h"
#endif
#include "rtc_base/checks.h"
#include "test/fuzzers/fuzz_data_helper.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {
  test::FuzzDataHelper fuzz_input(rtc::MakeArrayView(data, size));

  RtpPacketizer::PayloadSizeLimits limits;
  limits.max_payload_len = 1200;
  // Read uint8_t to be sure reduction_lens are much smaller than
  // max_payload_len and thus limits structure is valid.
  limits.first_packet_reduction_len = fuzz_input.ReadOrDefaultValue<uint8_t>(0);
  limits.last_packet_reduction_len = fuzz_input.ReadOrDefaultValue<uint8_t>(0);
  limits.single_packet_reduction_len =
      fuzz_input.ReadOrDefaultValue<uint8_t>(0);

  RTPVideoHeaderVP8 hdr_info;
  hdr_info.InitRTPVideoHeaderVP8();
#if WEBRTC_WEBKIT_BUILD
  hdr_info.nonReference = fuzz_input.ReadOrDefaultValue<uint8_t>(0) % 2;
#endif
  uint16_t picture_id = fuzz_input.ReadOrDefaultValue<uint16_t>(0);
  hdr_info.pictureId =
      picture_id >= 0x8000 ? kNoPictureId : picture_id & 0x7fff;
#if WEBRTC_WEBKIT_BUILD
  hdr_info.tl0PicIdx = fuzz_input.ReadOrDefaultValue<uint16_t>(kNoTl0PicIdx);
  hdr_info.temporalIdx = fuzz_input.ReadOrDefaultValue<uint8_t>(kNoTemporalIdx);
  hdr_info.layerSync = fuzz_input.ReadOrDefaultValue<uint8_t>(0) % 2;
  hdr_info.keyIdx = fuzz_input.ReadOrDefaultValue<int>(kNoKeyIdx);
  hdr_info.partitionId = fuzz_input.ReadOrDefaultValue<int>(0);
  hdr_info.beginningOfPartition = fuzz_input.ReadOrDefaultValue<uint8_t>(0) % 2;
#endif

  // Main function under test: RtpPacketizerVp8's constructor.
  RtpPacketizerVp8 packetizer(fuzz_input.ReadByteArray(fuzz_input.BytesLeft()),
                              limits, hdr_info);

  size_t num_packets = packetizer.NumPackets();
  if (num_packets == 0) {
    return;
  }
  // When packetization was successful, validate NextPacket function too.
  // While at it, check that packets respect the payload size limits.
#if WEBRTC_WEBKIT_BUILD
  // While at it, also depacketize the generated payloads.
  VideoRtpDepacketizerVp8 depacketizer;
#endif
  RtpPacketToSend rtp_packet(nullptr);
  // Single packet.
  if (num_packets == 1) {
    RTC_CHECK(packetizer.NextPacket(&rtp_packet));
    RTC_CHECK_LE(rtp_packet.payload_size(),
                 limits.max_payload_len - limits.single_packet_reduction_len);
#if WEBRTC_WEBKIT_BUILD
    depacketizer.Parse(rtp_packet.PayloadBuffer());
#endif
    return;
  }
  // First packet.
  RTC_CHECK(packetizer.NextPacket(&rtp_packet));
  RTC_CHECK_LE(rtp_packet.payload_size(),
               limits.max_payload_len - limits.first_packet_reduction_len);
#if WEBRTC_WEBKIT_BUILD
  depacketizer.Parse(rtp_packet.PayloadBuffer());
#endif
  // Middle packets.
  for (size_t i = 1; i < num_packets - 1; ++i) {
    RTC_CHECK(packetizer.NextPacket(&rtp_packet))
        << "Failed to get packet#" << i;
    RTC_CHECK_LE(rtp_packet.payload_size(), limits.max_payload_len)
        << "Packet #" << i << " exceeds it's limit";
#if WEBRTC_WEBKIT_BUILD
    depacketizer.Parse(rtp_packet.PayloadBuffer());
#endif
  }
  // Last packet.
  RTC_CHECK(packetizer.NextPacket(&rtp_packet));
  RTC_CHECK_LE(rtp_packet.payload_size(),
               limits.max_payload_len - limits.last_packet_reduction_len);
#if WEBRTC_WEBKIT_BUILD
  depacketizer.Parse(rtp_packet.PayloadBuffer());
#endif
}
}  // namespace webrtc
