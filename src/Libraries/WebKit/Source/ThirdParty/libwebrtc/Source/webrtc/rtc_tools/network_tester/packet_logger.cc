/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include "rtc_tools/network_tester/packet_logger.h"

#include <string>

#include "rtc_base/checks.h"

namespace webrtc {

PacketLogger::PacketLogger(const std::string& log_file_path)
    : packet_logger_stream_(log_file_path,
                            std::ios_base::out | std::ios_base::binary) {
  RTC_DCHECK(packet_logger_stream_.is_open());
  RTC_DCHECK(packet_logger_stream_.good());
}

PacketLogger::~PacketLogger() = default;

void PacketLogger::LogPacket(const NetworkTesterPacket& packet) {
  // The protobuffer message will be saved in the following format to the file:
  // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  // |      1 byte      |  X byte |      1 byte      | ... |  X byte |
  // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  // | Size of the next | proto   | Size of the next | ... | proto   |
  // | proto message    | message | proto message    |     | message |
  // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  std::string packet_data;
  packet.SerializeToString(&packet_data);
  RTC_DCHECK_LE(packet_data.length(), 255);
  RTC_DCHECK_GE(packet_data.length(), 0);
  char proto_size = packet_data.length();
  packet_logger_stream_.write(&proto_size, sizeof(proto_size));
  packet_logger_stream_.write(packet_data.data(), packet_data.length());
}

}  // namespace webrtc
