/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include "modules/audio_coding/neteq/tools/rtp_file_source.h"

#include <string.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/packet.h"
#include "modules/audio_coding/neteq/tools/packet_source.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/checks.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "test/rtp_file_reader.h"

namespace webrtc {
namespace test {

RtpFileSource* RtpFileSource::Create(absl::string_view file_name,
                                     std::optional<uint32_t> ssrc_filter) {
  RtpFileSource* source = new RtpFileSource(ssrc_filter);
  RTC_CHECK(source->OpenFile(file_name));
  return source;
}

bool RtpFileSource::ValidRtpDump(absl::string_view file_name) {
  std::unique_ptr<RtpFileReader> temp_file(
      RtpFileReader::Create(RtpFileReader::kRtpDump, file_name));
  return !!temp_file;
}

bool RtpFileSource::ValidPcap(absl::string_view file_name) {
  std::unique_ptr<RtpFileReader> temp_file(
      RtpFileReader::Create(RtpFileReader::kPcap, file_name));
  return !!temp_file;
}

RtpFileSource::~RtpFileSource() {}

bool RtpFileSource::RegisterRtpHeaderExtension(RTPExtensionType type,
                                               uint8_t id) {
  return rtp_header_extension_map_.RegisterByType(id, type);
}

std::unique_ptr<Packet> RtpFileSource::NextPacket() {
  while (true) {
    RtpPacket temp_packet;
    if (!rtp_reader_->NextPacket(&temp_packet)) {
      return NULL;
    }
    if (temp_packet.original_length == 0) {
      // May be an RTCP packet.
      // Read the next one.
      continue;
    }
    auto packet = std::make_unique<Packet>(
        rtc::CopyOnWriteBuffer(temp_packet.data, temp_packet.length),
        temp_packet.original_length, temp_packet.time_ms,
        &rtp_header_extension_map_);
    if (!packet->valid_header()) {
      continue;
    }
    if (filter_.test(packet->header().payloadType) ||
        (ssrc_filter_ && packet->header().ssrc != *ssrc_filter_)) {
      // This payload type should be filtered out. Continue to the next packet.
      continue;
    }
    return packet;
  }
}

RtpFileSource::RtpFileSource(std::optional<uint32_t> ssrc_filter)
    : PacketSource(), ssrc_filter_(ssrc_filter) {}

bool RtpFileSource::OpenFile(absl::string_view file_name) {
  rtp_reader_.reset(RtpFileReader::Create(RtpFileReader::kRtpDump, file_name));
  if (rtp_reader_)
    return true;
  rtp_reader_.reset(RtpFileReader::Create(RtpFileReader::kPcap, file_name));
  if (!rtp_reader_) {
    RTC_FATAL()
        << "Couldn't open input file as either a rtpdump or .pcap. Note "
        << "that .pcapng is not supported.";
  }
  return true;
}

}  // namespace test
}  // namespace webrtc
