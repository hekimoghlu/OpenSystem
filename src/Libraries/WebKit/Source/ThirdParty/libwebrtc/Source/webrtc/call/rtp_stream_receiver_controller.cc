/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include "call/rtp_stream_receiver_controller.h"

#include <cstdint>
#include <memory>

#include "api/sequence_checker.h"
#include "call/rtp_packet_sink_interface.h"
#include "call/rtp_stream_receiver_controller_interface.h"
#include "rtc_base/logging.h"

namespace webrtc {

RtpStreamReceiverController::Receiver::Receiver(
    RtpStreamReceiverController* controller,
    uint32_t ssrc,
    RtpPacketSinkInterface* sink)
    : controller_(controller), sink_(sink) {
  const bool sink_added = controller_->AddSink(ssrc, sink_);
  if (!sink_added) {
    RTC_LOG(LS_ERROR)
        << "RtpStreamReceiverController::Receiver::Receiver: Sink "
           "could not be added for SSRC="
        << ssrc << ".";
  }
}

RtpStreamReceiverController::Receiver::~Receiver() {
  // This may fail, if corresponding AddSink in the constructor failed.
  controller_->RemoveSink(sink_);
}

RtpStreamReceiverController::RtpStreamReceiverController() {}

RtpStreamReceiverController::~RtpStreamReceiverController() = default;

std::unique_ptr<RtpStreamReceiverInterface>
RtpStreamReceiverController::CreateReceiver(uint32_t ssrc,
                                            RtpPacketSinkInterface* sink) {
  return std::make_unique<Receiver>(this, ssrc, sink);
}

bool RtpStreamReceiverController::OnRtpPacket(const RtpPacketReceived& packet) {
  RTC_DCHECK_RUN_ON(&demuxer_sequence_);
  return demuxer_.OnRtpPacket(packet);
}

void RtpStreamReceiverController::OnRecoveredPacket(
    const RtpPacketReceived& packet) {
  RTC_DCHECK_RUN_ON(&demuxer_sequence_);
  demuxer_.OnRtpPacket(packet);
}

bool RtpStreamReceiverController::AddSink(uint32_t ssrc,
                                          RtpPacketSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(&demuxer_sequence_);
  return demuxer_.AddSink(ssrc, sink);
}

bool RtpStreamReceiverController::RemoveSink(
    const RtpPacketSinkInterface* sink) {
  RTC_DCHECK_RUN_ON(&demuxer_sequence_);
  return demuxer_.RemoveSink(sink);
}

}  // namespace webrtc
