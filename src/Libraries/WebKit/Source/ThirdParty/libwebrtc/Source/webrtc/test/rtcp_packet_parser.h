/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef TEST_RTCP_PACKET_PARSER_H_
#define TEST_RTCP_PACKET_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtcp_packet/app.h"
#include "modules/rtp_rtcp/source/rtcp_packet/bye.h"
#include "modules/rtp_rtcp/source/rtcp_packet/common_header.h"
#include "modules/rtp_rtcp/source/rtcp_packet/extended_reports.h"
#include "modules/rtp_rtcp/source/rtcp_packet/fir.h"
#include "modules/rtp_rtcp/source/rtcp_packet/loss_notification.h"
#include "modules/rtp_rtcp/source/rtcp_packet/nack.h"
#include "modules/rtp_rtcp/source/rtcp_packet/pli.h"
#include "modules/rtp_rtcp/source/rtcp_packet/rapid_resync_request.h"
#include "modules/rtp_rtcp/source/rtcp_packet/receiver_report.h"
#include "modules/rtp_rtcp/source/rtcp_packet/remb.h"
#include "modules/rtp_rtcp/source/rtcp_packet/sdes.h"
#include "modules/rtp_rtcp/source/rtcp_packet/sender_report.h"
#include "modules/rtp_rtcp/source/rtcp_packet/tmmbn.h"
#include "modules/rtp_rtcp/source/rtcp_packet/tmmbr.h"
#include "modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {
// Parse RTCP packet of given type. Assumes RTCP header is valid and that there
// is excatly one packet of correct type in the buffer.
template <typename Packet>
bool ParseSinglePacket(const uint8_t* buffer, size_t size, Packet* packet) {
  rtcp::CommonHeader header;
  RTC_CHECK(header.Parse(buffer, size));
  RTC_CHECK_EQ(size, header.NextPacket() - buffer);
  return packet->Parse(header);
}
// Same function, but takes raw buffer as single argument instead of pair.
template <typename Packet>
bool ParseSinglePacket(rtc::ArrayView<const uint8_t> buffer, Packet* packet) {
  return ParseSinglePacket(buffer.data(), buffer.size(), packet);
}

class RtcpPacketParser {
 public:
  // Keeps last parsed packet, count number of parsed packets of given type.
  template <typename TypedRtcpPacket>
  class PacketCounter : public TypedRtcpPacket {
   public:
    int num_packets() const { return num_packets_; }
    void Parse(const rtcp::CommonHeader& header) {
      if (TypedRtcpPacket::Parse(header))
        ++num_packets_;
    }
    bool Parse(const rtcp::CommonHeader& header, uint32_t* sender_ssrc) {
      const bool result = TypedRtcpPacket::Parse(header);
      if (result) {
        ++num_packets_;
        if (*sender_ssrc == 0)  // Use first sender ssrc in compound packet.
          *sender_ssrc = TypedRtcpPacket::sender_ssrc();
      }
      return result;
    }

   private:
    int num_packets_ = 0;
  };

  RtcpPacketParser();
  ~RtcpPacketParser();

  bool Parse(rtc::ArrayView<const uint8_t> packet);

  PacketCounter<rtcp::App>* app() { return &app_; }
  PacketCounter<rtcp::Bye>* bye() { return &bye_; }
  PacketCounter<rtcp::ExtendedReports>* xr() { return &xr_; }
  PacketCounter<rtcp::Fir>* fir() { return &fir_; }
  PacketCounter<rtcp::Nack>* nack() { return &nack_; }
  PacketCounter<rtcp::Pli>* pli() { return &pli_; }
  PacketCounter<rtcp::RapidResyncRequest>* rrr() { return &rrr_; }
  PacketCounter<rtcp::ReceiverReport>* receiver_report() {
    return &receiver_report_;
  }
  PacketCounter<rtcp::LossNotification>* loss_notification() {
    return &loss_notification_;
  }
  PacketCounter<rtcp::Remb>* remb() { return &remb_; }
  PacketCounter<rtcp::Sdes>* sdes() { return &sdes_; }
  PacketCounter<rtcp::SenderReport>* sender_report() { return &sender_report_; }
  PacketCounter<rtcp::Tmmbn>* tmmbn() { return &tmmbn_; }
  PacketCounter<rtcp::Tmmbr>* tmmbr() { return &tmmbr_; }
  PacketCounter<rtcp::TransportFeedback>* transport_feedback() {
    return &transport_feedback_;
  }
  uint32_t sender_ssrc() const { return sender_ssrc_; }
  size_t processed_rtcp_packets() const { return processed_rtcp_packets_; }

 private:
  PacketCounter<rtcp::App> app_;
  PacketCounter<rtcp::Bye> bye_;
  PacketCounter<rtcp::ExtendedReports> xr_;
  PacketCounter<rtcp::Fir> fir_;
  PacketCounter<rtcp::Nack> nack_;
  PacketCounter<rtcp::Pli> pli_;
  PacketCounter<rtcp::RapidResyncRequest> rrr_;
  PacketCounter<rtcp::ReceiverReport> receiver_report_;
  PacketCounter<rtcp::LossNotification> loss_notification_;
  PacketCounter<rtcp::Remb> remb_;
  PacketCounter<rtcp::Sdes> sdes_;
  PacketCounter<rtcp::SenderReport> sender_report_;
  PacketCounter<rtcp::Tmmbn> tmmbn_;
  PacketCounter<rtcp::Tmmbr> tmmbr_;
  PacketCounter<rtcp::TransportFeedback> transport_feedback_;
  uint32_t sender_ssrc_ = 0;
  size_t processed_rtcp_packets_ = 0;
};

}  // namespace test
}  // namespace webrtc
#endif  // TEST_RTCP_PACKET_PARSER_H_
