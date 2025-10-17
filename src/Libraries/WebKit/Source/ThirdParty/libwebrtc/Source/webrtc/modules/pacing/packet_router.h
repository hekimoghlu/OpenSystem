/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#ifndef MODULES_PACING_PACKET_ROUTER_H_
#define MODULES_PACING_PACKET_ROUTER_H_

#include <stddef.h>
#include <stdint.h>

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "api/array_view.h"
#include "api/sequence_checker.h"
#include "api/transport/network_types.h"
#include "api/units/data_size.h"
#include "modules/pacing/pacing_controller.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtcp_packet.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class RtpRtcpInterface;

// PacketRouter keeps track of rtp send modules to support the pacer.
// In addition, it handles feedback messages, which are sent on a send
// module if possible (sender report), otherwise on receive module
// (receiver report). For the latter case, we also keep track of the
// receive modules.
class PacketRouter : public PacingController::PacketSender {
 public:
  PacketRouter();
  ~PacketRouter() override;

  PacketRouter(const PacketRouter&) = delete;
  PacketRouter& operator=(const PacketRouter&) = delete;

  // Callback is invoked after pacing, before a packet is forwarded to the
  // sending rtp module.
  void RegisterNotifyBweCallback(
      absl::AnyInvocable<void(const RtpPacketToSend& packet,
                              const PacedPacketInfo& pacing_info)> callback);

  void AddSendRtpModule(RtpRtcpInterface* rtp_module, bool remb_candidate);
  void RemoveSendRtpModule(RtpRtcpInterface* rtp_module);

  bool SupportsRtxPayloadPadding() const;

  void AddReceiveRtpModule(RtcpFeedbackSenderInterface* rtcp_sender,
                           bool remb_candidate);
  void RemoveReceiveRtpModule(RtcpFeedbackSenderInterface* rtcp_sender);

  void SendPacket(std::unique_ptr<RtpPacketToSend> packet,
                  const PacedPacketInfo& cluster_info) override;
  std::vector<std::unique_ptr<RtpPacketToSend>> FetchFec() override;
  std::vector<std::unique_ptr<RtpPacketToSend>> GeneratePadding(
      DataSize size) override;
  void OnAbortedRetransmissions(
      uint32_t ssrc,
      rtc::ArrayView<const uint16_t> sequence_numbers) override;
  std::optional<uint32_t> GetRtxSsrcForMedia(uint32_t ssrc) const override;
  void OnBatchComplete() override;

  // Send REMB feedback.
  void SendRemb(int64_t bitrate_bps, std::vector<uint32_t> ssrcs);

  // Sends `packets` in one or more IP packets.
  void SendCombinedRtcpPacket(
      std::vector<std::unique_ptr<rtcp::RtcpPacket>> packets);

 private:
  void AddRembModuleCandidate(RtcpFeedbackSenderInterface* candidate_module,
                              bool media_sender);
  void MaybeRemoveRembModuleCandidate(
      RtcpFeedbackSenderInterface* candidate_module,
      bool media_sender);
  void UnsetActiveRembModule();
  void DetermineActiveRembModule();
  void AddSendRtpModuleToMap(RtpRtcpInterface* rtp_module, uint32_t ssrc);
  void RemoveSendRtpModuleFromMap(uint32_t ssrc);

  SequenceChecker thread_checker_;
  // Ssrc to RtpRtcpInterface module;
  std::unordered_map<uint32_t, RtpRtcpInterface*> send_modules_map_
      RTC_GUARDED_BY(thread_checker_);
  std::list<RtpRtcpInterface*> send_modules_list_
      RTC_GUARDED_BY(thread_checker_);
  // The last module used to send media.
  RtpRtcpInterface* last_send_module_ RTC_GUARDED_BY(thread_checker_);
  // Rtcp modules of the rtp receivers.
  std::vector<RtcpFeedbackSenderInterface*> rtcp_feedback_senders_
      RTC_GUARDED_BY(thread_checker_);

  // Candidates for the REMB module can be RTP sender/receiver modules, with
  // the sender modules taking precedence.
  std::vector<RtcpFeedbackSenderInterface*> sender_remb_candidates_
      RTC_GUARDED_BY(thread_checker_);
  std::vector<RtcpFeedbackSenderInterface*> receiver_remb_candidates_
      RTC_GUARDED_BY(thread_checker_);
  RtcpFeedbackSenderInterface* active_remb_module_
      RTC_GUARDED_BY(thread_checker_);

  uint64_t transport_seq_ RTC_GUARDED_BY(thread_checker_);
  absl::AnyInvocable<void(RtpPacketToSend& packet,
                          const PacedPacketInfo& pacing_info)>
      notify_bwe_callback_ RTC_GUARDED_BY(thread_checker_) = nullptr;

  std::vector<std::unique_ptr<RtpPacketToSend>> pending_fec_packets_
      RTC_GUARDED_BY(thread_checker_);
  std::set<RtpRtcpInterface*> modules_used_in_current_batch_
      RTC_GUARDED_BY(thread_checker_);
};
}  // namespace webrtc
#endif  // MODULES_PACING_PACKET_ROUTER_H_
