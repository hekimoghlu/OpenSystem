/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#ifndef RTC_TOOLS_NETWORK_TESTER_PACKET_LOGGER_H_
#define RTC_TOOLS_NETWORK_TESTER_PACKET_LOGGER_H_

#include <fstream>
#include <string>

#ifdef WEBRTC_NETWORK_TESTER_PROTO
#include "rtc_tools/network_tester/network_tester_packet.pb.h"
using webrtc::network_tester::packet::NetworkTesterPacket;
#else
class NetworkTesterPacket;
#endif  // WEBRTC_NETWORK_TESTER_PROTO

namespace webrtc {

class PacketLogger {
 public:
  explicit PacketLogger(const std::string& log_file_path);
  ~PacketLogger();

  PacketLogger(const PacketLogger&) = delete;
  PacketLogger& operator=(const PacketLogger&) = delete;

  void LogPacket(const NetworkTesterPacket& packet);

 private:
  std::ofstream packet_logger_stream_;
};

}  // namespace webrtc

#endif  // RTC_TOOLS_NETWORK_TESTER_PACKET_LOGGER_H_
