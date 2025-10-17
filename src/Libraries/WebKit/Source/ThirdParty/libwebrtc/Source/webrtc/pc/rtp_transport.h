/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#ifndef PC_RTP_TRANSPORT_H_
#define PC_RTP_TRANSPORT_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>

#include "api/field_trials_view.h"
#include "api/task_queue/pending_task_safety_flag.h"
#include "api/units/timestamp.h"
#include "call/rtp_demuxer.h"
#include "call/video_receive_stream.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "p2p/base/packet_transport_internal.h"
#include "pc/rtp_transport_internal.h"
#include "pc/session_description.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/network/received_packet.h"
#include "rtc_base/network/sent_packet.h"
#include "rtc_base/network_route.h"
#include "rtc_base/socket.h"

namespace rtc {

class CopyOnWriteBuffer;
struct PacketOptions;
class PacketTransportInternal;

}  // namespace rtc

namespace webrtc {

class RtpTransport : public RtpTransportInternal {
 public:
  RtpTransport(const RtpTransport&) = delete;
  RtpTransport& operator=(const RtpTransport&) = delete;

  RtpTransport(bool rtcp_mux_enabled, const FieldTrialsView& field_trials)
      : set_ready_to_send_false_if_send_fail_(
            field_trials.IsEnabled("WebRTC-SetReadyToSendFalseIfSendFail")),
        rtcp_mux_enabled_(rtcp_mux_enabled) {}

  bool rtcp_mux_enabled() const override { return rtcp_mux_enabled_; }
  void SetRtcpMuxEnabled(bool enable) override;

  const std::string& transport_name() const override;

  int SetRtpOption(rtc::Socket::Option opt, int value) override;
  int SetRtcpOption(rtc::Socket::Option opt, int value) override;

  rtc::PacketTransportInternal* rtp_packet_transport() const {
    return rtp_packet_transport_;
  }
  void SetRtpPacketTransport(rtc::PacketTransportInternal* rtp);

  rtc::PacketTransportInternal* rtcp_packet_transport() const {
    return rtcp_packet_transport_;
  }
  void SetRtcpPacketTransport(rtc::PacketTransportInternal* rtcp);

  bool IsReadyToSend() const override { return ready_to_send_; }

  bool IsWritable(bool rtcp) const override;

  bool SendRtpPacket(rtc::CopyOnWriteBuffer* packet,
                     const rtc::PacketOptions& options,
                     int flags) override;

  bool SendRtcpPacket(rtc::CopyOnWriteBuffer* packet,
                      const rtc::PacketOptions& options,
                      int flags) override;

  bool IsSrtpActive() const override { return false; }

  void UpdateRtpHeaderExtensionMap(
      const cricket::RtpHeaderExtensions& header_extensions) override;

  bool RegisterRtpDemuxerSink(const RtpDemuxerCriteria& criteria,
                              RtpPacketSinkInterface* sink) override;

  bool UnregisterRtpDemuxerSink(RtpPacketSinkInterface* sink) override;

 protected:
  // These methods will be used in the subclasses.
  void DemuxPacket(rtc::CopyOnWriteBuffer packet,
                   Timestamp arrival_time,
                   rtc::EcnMarking ecn);

  bool SendPacket(bool rtcp,
                  rtc::CopyOnWriteBuffer* packet,
                  const rtc::PacketOptions& options,
                  int flags);
  flat_set<uint32_t> GetSsrcsForSink(RtpPacketSinkInterface* sink);

  // Overridden by SrtpTransport.
  virtual void OnNetworkRouteChanged(
      std::optional<rtc::NetworkRoute> network_route);
  virtual void OnRtpPacketReceived(const rtc::ReceivedPacket& packet);
  virtual void OnRtcpPacketReceived(const rtc::ReceivedPacket& packet);
  // Overridden by SrtpTransport and DtlsSrtpTransport.
  virtual void OnWritableState(rtc::PacketTransportInternal* packet_transport);

 private:
  void OnReadyToSend(rtc::PacketTransportInternal* transport);
  void OnSentPacket(rtc::PacketTransportInternal* packet_transport,
                    const rtc::SentPacket& sent_packet);
  void OnReadPacket(rtc::PacketTransportInternal* transport,
                    const rtc::ReceivedPacket& received_packet);

  // Updates "ready to send" for an individual channel and fires
  // SignalReadyToSend.
  void SetReadyToSend(bool rtcp, bool ready);

  void MaybeSignalReadyToSend();

  bool IsTransportWritable();

  const bool set_ready_to_send_false_if_send_fail_;
  bool rtcp_mux_enabled_;

  rtc::PacketTransportInternal* rtp_packet_transport_ = nullptr;
  rtc::PacketTransportInternal* rtcp_packet_transport_ = nullptr;

  bool ready_to_send_ = false;
  bool rtp_ready_to_send_ = false;
  bool rtcp_ready_to_send_ = false;

  RtpDemuxer rtp_demuxer_;

  // Used for identifying the MID for RtpDemuxer.
  RtpHeaderExtensionMap header_extension_map_;
  // Guard against recursive "ready to send" signals
  bool processing_ready_to_send_ = false;
  bool processing_sent_packet_ = false;
  ScopedTaskSafety safety_;
};

}  // namespace webrtc

#endif  // PC_RTP_TRANSPORT_H_
