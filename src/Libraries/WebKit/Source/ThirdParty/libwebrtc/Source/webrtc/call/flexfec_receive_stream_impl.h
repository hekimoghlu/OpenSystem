/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#ifndef CALL_FLEXFEC_RECEIVE_STREAM_IMPL_H_
#define CALL_FLEXFEC_RECEIVE_STREAM_IMPL_H_

#include <cstdint>
#include <memory>

#include "api/environment/environment.h"
#include "api/rtp_headers.h"
#include "api/sequence_checker.h"
#include "call/flexfec_receive_stream.h"
#include "call/rtp_packet_sink_interface.h"
#include "modules/rtp_rtcp/source/rtp_rtcp_impl2.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class FlexfecReceiver;
class ReceiveStatistics;
class RecoveredPacketReceiver;
class RtcpRttStats;
class RtpPacketReceived;
class RtpStreamReceiverControllerInterface;
class RtpStreamReceiverInterface;

class FlexfecReceiveStreamImpl : public FlexfecReceiveStream {
 public:
  FlexfecReceiveStreamImpl(const Environment& env,
                           Config config,
                           RecoveredPacketReceiver* recovered_packet_receiver,
                           RtcpRttStats* rtt_stats);
  // Destruction happens on the worker thread. Prior to destruction the caller
  // must ensure that a registration with the transport has been cleared. See
  // `RegisterWithTransport` for details.
  // TODO(tommi): As a further improvement to this, performing the full
  // destruction on the network thread could be made the default.
  ~FlexfecReceiveStreamImpl() override;

  // Called on the network thread to register/unregister with the network
  // transport.
  void RegisterWithTransport(
      RtpStreamReceiverControllerInterface* receiver_controller);
  // If registration has previously been done (via `RegisterWithTransport`) then
  // `UnregisterFromTransport` must be called prior to destruction, on the
  // network thread.
  void UnregisterFromTransport();

  // RtpPacketSinkInterface.
  void OnRtpPacket(const RtpPacketReceived& packet) override;

  void SetPayloadType(int payload_type) override;
  int payload_type() const override;

  // Updates the `rtp_video_stream_receiver_`'s `local_ssrc` when the default
  // sender has been created, changed or removed.
  void SetLocalSsrc(uint32_t local_ssrc);

  uint32_t remote_ssrc() const { return remote_ssrc_; }

  void SetRtcpMode(RtcpMode mode) override {
    RTC_DCHECK_RUN_ON(&packet_sequence_checker_);
    rtp_rtcp_.SetRTCPStatus(mode);
  }

  const ReceiveStatistics* GetStats() const override {
    return rtp_receive_statistics_.get();
  }

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker packet_sequence_checker_;

  const uint32_t remote_ssrc_;

  // `payload_type_` is initially set to -1, indicating that FlexFec is
  // disabled.
  int payload_type_ RTC_GUARDED_BY(packet_sequence_checker_) = -1;

  // Erasure code interfacing.
  const std::unique_ptr<FlexfecReceiver> receiver_;

  // RTCP reporting.
  const std::unique_ptr<ReceiveStatistics> rtp_receive_statistics_;
  ModuleRtpRtcpImpl2 rtp_rtcp_;

  std::unique_ptr<RtpStreamReceiverInterface> rtp_stream_receiver_
      RTC_GUARDED_BY(packet_sequence_checker_);
};

}  // namespace webrtc

#endif  // CALL_FLEXFEC_RECEIVE_STREAM_IMPL_H_
