/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "call/flexfec_receive_stream_impl.h"

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>

#include "api/array_view.h"
#include "api/environment/environment.h"
#include "api/sequence_checker.h"
#include "call/flexfec_receive_stream.h"
#include "call/rtp_stream_receiver_controller_interface.h"
#include "modules/rtp_rtcp/include/flexfec_receiver.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

std::string FlexfecReceiveStream::Config::ToString() const {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  ss << "{payload_type: " << payload_type;
  ss << ", remote_ssrc: " << rtp.remote_ssrc;
  ss << ", local_ssrc: " << rtp.local_ssrc;
  ss << ", protected_media_ssrcs: [";
  size_t i = 0;
  for (; i + 1 < protected_media_ssrcs.size(); ++i)
    ss << protected_media_ssrcs[i] << ", ";
  if (!protected_media_ssrcs.empty())
    ss << protected_media_ssrcs[i];
  ss << "}";
  return ss.str();
}

bool FlexfecReceiveStream::Config::IsCompleteAndEnabled() const {
  // Check if FlexFEC is enabled.
  if (payload_type < 0)
    return false;
  // Do we have the necessary SSRC information?
  if (rtp.remote_ssrc == 0)
    return false;
  // TODO(brandtr): Update this check when we support multistream protection.
  if (protected_media_ssrcs.size() != 1u)
    return false;
  return true;
}

namespace {

// TODO(brandtr): Update this function when we support multistream protection.
std::unique_ptr<FlexfecReceiver> MaybeCreateFlexfecReceiver(
    Clock* clock,
    const FlexfecReceiveStream::Config& config,
    RecoveredPacketReceiver* recovered_packet_receiver) {
  if (config.payload_type < 0) {
    RTC_LOG(LS_WARNING)
        << "Invalid FlexFEC payload type given. "
           "This FlexfecReceiveStream will therefore be useless.";
    return nullptr;
  }
  RTC_DCHECK_GE(config.payload_type, 0);
  RTC_DCHECK_LE(config.payload_type, 127);
  if (config.rtp.remote_ssrc == 0) {
    RTC_LOG(LS_WARNING)
        << "Invalid FlexFEC SSRC given. "
           "This FlexfecReceiveStream will therefore be useless.";
    return nullptr;
  }
  if (config.protected_media_ssrcs.empty()) {
    RTC_LOG(LS_WARNING)
        << "No protected media SSRC supplied. "
           "This FlexfecReceiveStream will therefore be useless.";
    return nullptr;
  }

  if (config.protected_media_ssrcs.size() > 1) {
    RTC_LOG(LS_WARNING)
        << "The supplied FlexfecConfig contained multiple protected "
           "media streams, but our implementation currently only "
           "supports protecting a single media stream. "
           "To avoid confusion, disabling FlexFEC completely.";
    return nullptr;
  }
  RTC_DCHECK_EQ(1U, config.protected_media_ssrcs.size());
  return std::unique_ptr<FlexfecReceiver>(new FlexfecReceiver(
      clock, config.rtp.remote_ssrc, config.protected_media_ssrcs[0],
      recovered_packet_receiver));
}

}  // namespace

FlexfecReceiveStreamImpl::FlexfecReceiveStreamImpl(
    const Environment& env,
    Config config,
    RecoveredPacketReceiver* recovered_packet_receiver,
    RtcpRttStats* rtt_stats)
    : remote_ssrc_(config.rtp.remote_ssrc),
      payload_type_(config.payload_type),
      receiver_(MaybeCreateFlexfecReceiver(&env.clock(),
                                           config,
                                           recovered_packet_receiver)),
      rtp_receive_statistics_(ReceiveStatistics::Create(&env.clock())),
      rtp_rtcp_(env,
                {.audio = false,
                 .receiver_only = true,
                 .receive_statistics = rtp_receive_statistics_.get(),
                 .outgoing_transport = config.rtcp_send_transport,
                 .rtt_stats = rtt_stats,
                 .local_media_ssrc = config.rtp.local_ssrc}) {
  RTC_LOG(LS_INFO) << "FlexfecReceiveStreamImpl: " << config.ToString();
  RTC_DCHECK_GE(payload_type_, -1);

  packet_sequence_checker_.Detach();

  // RTCP reporting.
  rtp_rtcp_.SetRTCPStatus(config.rtcp_mode);
}

FlexfecReceiveStreamImpl::~FlexfecReceiveStreamImpl() {
  RTC_DLOG(LS_INFO) << "~FlexfecReceiveStreamImpl: ssrc: " << remote_ssrc_;
}

void FlexfecReceiveStreamImpl::RegisterWithTransport(
    RtpStreamReceiverControllerInterface* receiver_controller) {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  RTC_DCHECK(!rtp_stream_receiver_);

  if (!receiver_)
    return;

  // TODO(nisse): OnRtpPacket in this class delegates all real work to
  // `receiver_`. So maybe we don't need to implement RtpPacketSinkInterface
  // here at all, we'd then delete the OnRtpPacket method and instead register
  // `receiver_` as the RtpPacketSinkInterface for this stream.
  rtp_stream_receiver_ =
      receiver_controller->CreateReceiver(remote_ssrc(), this);
}

void FlexfecReceiveStreamImpl::UnregisterFromTransport() {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  rtp_stream_receiver_.reset();
}

void FlexfecReceiveStreamImpl::OnRtpPacket(const RtpPacketReceived& packet) {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  if (!receiver_)
    return;

  receiver_->OnRtpPacket(packet);

  // Do not report media packets in the RTCP RRs generated by `rtp_rtcp_`.
  if (packet.Ssrc() == remote_ssrc()) {
    rtp_receive_statistics_->OnRtpPacket(packet);
  }
}

void FlexfecReceiveStreamImpl::SetPayloadType(int payload_type) {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  RTC_DCHECK_GE(payload_type, -1);
  payload_type_ = payload_type;
}

int FlexfecReceiveStreamImpl::payload_type() const {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  return payload_type_;
}

void FlexfecReceiveStreamImpl::SetLocalSsrc(uint32_t local_ssrc) {
  RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
  if (local_ssrc == rtp_rtcp_.local_media_ssrc())
    return;

  rtp_rtcp_.SetLocalSsrc(local_ssrc);
}

}  // namespace webrtc
