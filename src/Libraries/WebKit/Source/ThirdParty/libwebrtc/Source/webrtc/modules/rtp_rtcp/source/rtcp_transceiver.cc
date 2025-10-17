/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "modules/rtp_rtcp/source/rtcp_transceiver.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"
#include "rtc_base/checks.h"
#include "rtc_base/event.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

RtcpTransceiver::RtcpTransceiver(const RtcpTransceiverConfig& config)
    : clock_(config.clock),
      task_queue_(config.task_queue),
      rtcp_transceiver_(std::make_unique<RtcpTransceiverImpl>(config)) {
  RTC_DCHECK(task_queue_);
}

RtcpTransceiver::~RtcpTransceiver() {
  if (!rtcp_transceiver_)
    return;
  auto rtcp_transceiver = std::move(rtcp_transceiver_);
  task_queue_->PostTask([rtcp_transceiver = std::move(rtcp_transceiver)] {
    rtcp_transceiver->StopPeriodicTask();
  });
  RTC_DCHECK(!rtcp_transceiver_);
}

void RtcpTransceiver::Stop(absl::AnyInvocable<void() &&> on_destroyed) {
  RTC_DCHECK(rtcp_transceiver_);
  auto rtcp_transceiver = std::move(rtcp_transceiver_);
  absl::Cleanup cleanup = std::move(on_destroyed);
  task_queue_->PostTask(
      [rtcp_transceiver = std::move(rtcp_transceiver),
       cleanup = std::move(cleanup)] { rtcp_transceiver->StopPeriodicTask(); });
  RTC_DCHECK(!rtcp_transceiver_);
}

void RtcpTransceiver::AddMediaReceiverRtcpObserver(
    uint32_t remote_ssrc,
    MediaReceiverRtcpObserver* observer) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr, remote_ssrc, observer] {
    ptr->AddMediaReceiverRtcpObserver(remote_ssrc, observer);
  });
}

void RtcpTransceiver::RemoveMediaReceiverRtcpObserver(
    uint32_t remote_ssrc,
    MediaReceiverRtcpObserver* observer,
    absl::AnyInvocable<void() &&> on_removed) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  absl::Cleanup cleanup = std::move(on_removed);
  task_queue_->PostTask(
      [ptr, remote_ssrc, observer, cleanup = std::move(cleanup)] {
        ptr->RemoveMediaReceiverRtcpObserver(remote_ssrc, observer);
      });
}

void RtcpTransceiver::SetReadyToSend(bool ready) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr, ready] { ptr->SetReadyToSend(ready); });
}

void RtcpTransceiver::ReceivePacket(rtc::CopyOnWriteBuffer packet) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  Timestamp now = clock_->CurrentTime();
  task_queue_->PostTask(
      [ptr, packet, now] { ptr->ReceivePacket(packet, now); });
}

void RtcpTransceiver::SendCompoundPacket() {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr] { ptr->SendCompoundPacket(); });
}

void RtcpTransceiver::SetRemb(int64_t bitrate_bps,
                              std::vector<uint32_t> ssrcs) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr, bitrate_bps, ssrcs = std::move(ssrcs)]() mutable {
    ptr->SetRemb(bitrate_bps, std::move(ssrcs));
  });
}

void RtcpTransceiver::UnsetRemb() {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr] { ptr->UnsetRemb(); });
}

void RtcpTransceiver::SendCombinedRtcpPacket(
    std::vector<std::unique_ptr<rtcp::RtcpPacket>> rtcp_packets) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask(
      [ptr, rtcp_packets = std::move(rtcp_packets)]() mutable {
        ptr->SendCombinedRtcpPacket(std::move(rtcp_packets));
      });
}

void RtcpTransceiver::SendNack(uint32_t ssrc,
                               std::vector<uint16_t> sequence_numbers) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask(
      [ptr, ssrc, sequence_numbers = std::move(sequence_numbers)]() mutable {
        ptr->SendNack(ssrc, std::move(sequence_numbers));
      });
}

void RtcpTransceiver::SendPictureLossIndication(uint32_t ssrc) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr, ssrc] { ptr->SendPictureLossIndication(ssrc); });
}

void RtcpTransceiver::SendFullIntraRequest(std::vector<uint32_t> ssrcs) {
  return SendFullIntraRequest(std::move(ssrcs), true);
}

void RtcpTransceiver::SendFullIntraRequest(std::vector<uint32_t> ssrcs,
                                           bool new_request) {
  RTC_CHECK(rtcp_transceiver_);
  RtcpTransceiverImpl* ptr = rtcp_transceiver_.get();
  task_queue_->PostTask([ptr, ssrcs = std::move(ssrcs), new_request] {
    ptr->SendFullIntraRequest(ssrcs, new_request);
  });
}

}  // namespace webrtc
