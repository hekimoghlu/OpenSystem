/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include <stdio.h>

#include <memory>

#include "modules/remote_bitrate_estimator/tools/bwe_rtp.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "modules/rtp_rtcp/source/rtp_packet.h"
#include "rtc_base/strings/string_builder.h"
#include "test/rtp_file_reader.h"

int main(int argc, char* argv[]) {
  std::unique_ptr<webrtc::test::RtpFileReader> reader;
  webrtc::RtpHeaderExtensionMap rtp_header_extensions;
  if (!ParseArgsAndSetupRtpReader(argc, argv, reader, rtp_header_extensions)) {
    return -1;
  }

  bool arrival_time_only = (argc >= 5 && strncmp(argv[4], "-t", 2) == 0);

  fprintf(stdout,
          "seqnum timestamp ts_offset abs_sendtime recvtime "
          "markerbit ssrc size original_size\n");
  int packet_counter = 0;
  int non_zero_abs_send_time = 0;
  int non_zero_ts_offsets = 0;
  webrtc::test::RtpPacket packet;
  while (reader->NextPacket(&packet)) {
    webrtc::RtpPacket header(&rtp_header_extensions);
    header.Parse(packet.data, packet.length);
    uint32_t abs_send_time = 0;
    if (header.GetExtension<webrtc::AbsoluteSendTime>(&abs_send_time) &&
        abs_send_time != 0)
      ++non_zero_abs_send_time;
    int32_t toffset = 0;
    if (header.GetExtension<webrtc::TransmissionOffset>(&toffset) &&
        toffset != 0)
      ++non_zero_ts_offsets;
    if (arrival_time_only) {
      rtc::StringBuilder ss;
      ss << static_cast<int64_t>(packet.time_ms) * 1000000;
      fprintf(stdout, "%s\n", ss.str().c_str());
    } else {
      fprintf(stdout, "%u %u %d %u %u %d %u %zu %zu\n", header.SequenceNumber(),
              header.Timestamp(), toffset, abs_send_time, packet.time_ms,
              header.Marker(), header.Ssrc(), packet.length,
              packet.original_length);
    }
    ++packet_counter;
  }
  fprintf(stderr, "Parsed %d packets\n", packet_counter);
  fprintf(stderr, "Packets with non-zero absolute send time: %d\n",
          non_zero_abs_send_time);
  fprintf(stderr, "Packets with non-zero timestamp offset: %d\n",
          non_zero_ts_offsets);
  return 0;
}
