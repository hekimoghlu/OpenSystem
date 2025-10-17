/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include "net/dcsctp/public/text_pcap_packet_observer.h"

#include "api/array_view.h"
#include "net/dcsctp/public/types.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

void TextPcapPacketObserver::OnSentPacket(
    dcsctp::TimeMs now,
    rtc::ArrayView<const uint8_t> payload) {
  PrintPacket("O ", name_, now, payload);
}

void TextPcapPacketObserver::OnReceivedPacket(
    dcsctp::TimeMs now,
    rtc::ArrayView<const uint8_t> payload) {
  PrintPacket("I ", name_, now, payload);
}

void TextPcapPacketObserver::PrintPacket(
    absl::string_view prefix,
    absl::string_view socket_name,
    dcsctp::TimeMs now,
    rtc::ArrayView<const uint8_t> payload) {
  rtc::StringBuilder s;
  s << "\n" << prefix;
  int64_t remaining = *now % (24 * 60 * 60 * 1000);
  int hours = remaining / (60 * 60 * 1000);
  remaining = remaining % (60 * 60 * 1000);
  int minutes = remaining / (60 * 1000);
  remaining = remaining % (60 * 1000);
  int seconds = remaining / 1000;
  int ms = remaining % 1000;
  s.AppendFormat("%02d:%02d:%02d.%03d", hours, minutes, seconds, ms);
  s << " 0000";
  for (uint8_t byte : payload) {
    s.AppendFormat(" %02x", byte);
  }
  s << " # SCTP_PACKET " << socket_name;
  RTC_LOG(LS_VERBOSE) << s.str();
}

}  // namespace dcsctp
