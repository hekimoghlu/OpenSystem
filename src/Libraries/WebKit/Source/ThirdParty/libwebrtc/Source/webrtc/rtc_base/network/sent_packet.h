/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef RTC_BASE_NETWORK_SENT_PACKET_H_
#define RTC_BASE_NETWORK_SENT_PACKET_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "rtc_base/system/rtc_export.h"

namespace rtc {

enum class PacketType {
  kUnknown,
  kData,
  kIceConnectivityCheck,
  kIceConnectivityCheckResponse,
  kStunMessage,
  kTurnMessage,
};

enum class PacketInfoProtocolType {
  kUnknown,
  kUdp,
  kTcp,
  kSsltcp,
  kTls,
};

struct RTC_EXPORT PacketInfo {
  PacketInfo();
  PacketInfo(const PacketInfo& info);
  ~PacketInfo();

  bool included_in_feedback = false;
  bool included_in_allocation = false;
  // `is_media` is true if this is an audio or video packet, excluding
  // retransmissions.
  bool is_media = false;
  PacketType packet_type = PacketType::kUnknown;
  PacketInfoProtocolType protocol = PacketInfoProtocolType::kUnknown;
  // A unique id assigned by the network manager, and std::nullopt if not set.
  std::optional<uint16_t> network_id;
  size_t packet_size_bytes = 0;
  size_t turn_overhead_bytes = 0;
  size_t ip_overhead_bytes = 0;
};

struct RTC_EXPORT SentPacket {
  SentPacket();
  SentPacket(int64_t packet_id, int64_t send_time_ms);
  SentPacket(int64_t packet_id,
             int64_t send_time_ms,
             const rtc::PacketInfo& info);

  int64_t packet_id = -1;
  int64_t send_time_ms = -1;
  rtc::PacketInfo info;
};

}  // namespace rtc

#endif  // RTC_BASE_NETWORK_SENT_PACKET_H_
