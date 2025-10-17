/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_RTP_FILE_SOURCE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_RTP_FILE_SOURCE_H_

#include <stdio.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/packet_source.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {

namespace test {

class RtpFileReader;

class RtpFileSource : public PacketSource {
 public:
  // Creates an RtpFileSource reading from `file_name`. If the file cannot be
  // opened, or has the wrong format, NULL will be returned.
  static RtpFileSource* Create(
      absl::string_view file_name,
      std::optional<uint32_t> ssrc_filter = std::nullopt);

  // Checks whether a files is a valid RTP dump or PCAP (Wireshark) file.
  static bool ValidRtpDump(absl::string_view file_name);
  static bool ValidPcap(absl::string_view file_name);

  ~RtpFileSource() override;

  RtpFileSource(const RtpFileSource&) = delete;
  RtpFileSource& operator=(const RtpFileSource&) = delete;

  // Registers an RTP header extension and binds it to `id`.
  virtual bool RegisterRtpHeaderExtension(RTPExtensionType type, uint8_t id);

  std::unique_ptr<Packet> NextPacket() override;

 private:
  static const int kFirstLineLength = 40;
  static const int kRtpFileHeaderSize = 4 + 4 + 4 + 2 + 2;
  static const size_t kPacketHeaderSize = 8;

  explicit RtpFileSource(std::optional<uint32_t> ssrc_filter);

  bool OpenFile(absl::string_view file_name);

  std::unique_ptr<RtpFileReader> rtp_reader_;
  const std::optional<uint32_t> ssrc_filter_;
  RtpHeaderExtensionMap rtp_header_extension_map_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_RTP_FILE_SOURCE_H_
