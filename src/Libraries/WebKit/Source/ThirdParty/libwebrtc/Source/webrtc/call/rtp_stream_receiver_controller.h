/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#ifndef CALL_RTP_STREAM_RECEIVER_CONTROLLER_H_
#define CALL_RTP_STREAM_RECEIVER_CONTROLLER_H_

#include <cstdint>
#include <memory>

#include "api/sequence_checker.h"
#include "call/rtp_demuxer.h"
#include "call/rtp_stream_receiver_controller_interface.h"
#include "modules/rtp_rtcp/include/recovered_packet_receiver.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class RtpPacketReceived;

// This class represents the RTP receive parsing and demuxing, for a
// single RTP session.
// TODO(bugs.webrtc.org/7135): Add RTCP processing, we should aim to terminate
// RTCP and not leave any RTCP processing to individual receive streams.
class RtpStreamReceiverController : public RtpStreamReceiverControllerInterface,
                                    public RecoveredPacketReceiver {
 public:
  RtpStreamReceiverController();
  ~RtpStreamReceiverController() override;

  // Implements RtpStreamReceiverControllerInterface.
  std::unique_ptr<RtpStreamReceiverInterface> CreateReceiver(
      uint32_t ssrc,
      RtpPacketSinkInterface* sink) override;

  // TODO(bugs.webrtc.org/7135): Not yet responsible for parsing.
  bool OnRtpPacket(const RtpPacketReceived& packet);

  // Implements RecoveredPacketReceiver.
  // Responsible for demuxing recovered FLEXFEC packets.
  void OnRecoveredPacket(const RtpPacketReceived& packet) override;

 private:
  class Receiver : public RtpStreamReceiverInterface {
   public:
    Receiver(RtpStreamReceiverController* controller,
             uint32_t ssrc,
             RtpPacketSinkInterface* sink);

    ~Receiver() override;

   private:
    RtpStreamReceiverController* const controller_;
    RtpPacketSinkInterface* const sink_;
  };

  // Thread-safe wrappers for the corresponding RtpDemuxer methods.
  bool AddSink(uint32_t ssrc, RtpPacketSinkInterface* sink);
  bool RemoveSink(const RtpPacketSinkInterface* sink);

  // TODO(bugs.webrtc.org/11993): We expect construction and all methods to be
  // called on the same thread/tq. Currently this is the worker thread
  // (including OnRtpPacket) but a more natural fit would be the network thread.
  // Using a sequence checker to ensure that usage is correct but at the same
  // time not require a specific thread/tq, an instance of this class + the
  // associated functionality should be easily moved from one execution context
  // to another (i.e. when network packets don't hop to the worker thread inside
  // of Call).
  SequenceChecker demuxer_sequence_;
  // At this level the demuxer is only configured to demux by SSRC, so don't
  // worry about MIDs (MIDs are handled by upper layers).
  RtpDemuxer demuxer_ RTC_GUARDED_BY(&demuxer_sequence_){false /*use_mid*/};
};

}  // namespace webrtc

#endif  // CALL_RTP_STREAM_RECEIVER_CONTROLLER_H_
