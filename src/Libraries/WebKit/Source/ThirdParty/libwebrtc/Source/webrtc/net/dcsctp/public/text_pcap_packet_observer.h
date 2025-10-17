/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#ifndef NET_DCSCTP_PUBLIC_TEXT_PCAP_PACKET_OBSERVER_H_
#define NET_DCSCTP_PUBLIC_TEXT_PCAP_PACKET_OBSERVER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/public/packet_observer.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// Print outs all sent and received packets to the logs, at LS_VERBOSE severity.
class TextPcapPacketObserver : public dcsctp::PacketObserver {
 public:
  explicit TextPcapPacketObserver(absl::string_view name) : name_(name) {}

  // Implementation of `dcsctp::PacketObserver`.
  void OnSentPacket(dcsctp::TimeMs now,
                    rtc::ArrayView<const uint8_t> payload) override;

  void OnReceivedPacket(dcsctp::TimeMs now,
                        rtc::ArrayView<const uint8_t> payload) override;

  // Prints a packet to the log. Exposed to allow it to be used in compatibility
  // tests suites that don't use PacketObserver.
  static void PrintPacket(absl::string_view prefix,
                          absl::string_view socket_name,
                          dcsctp::TimeMs now,
                          rtc::ArrayView<const uint8_t> payload);

 private:
  const std::string name_;
};

}  // namespace dcsctp
#endif  // NET_DCSCTP_PUBLIC_TEXT_PCAP_PACKET_OBSERVER_H_
