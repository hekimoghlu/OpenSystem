/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#ifndef CALL_RTP_STREAM_RECEIVER_CONTROLLER_INTERFACE_H_
#define CALL_RTP_STREAM_RECEIVER_CONTROLLER_INTERFACE_H_

#include <cstdint>
#include <memory>

#include "call/rtp_packet_sink_interface.h"

namespace webrtc {

// An RtpStreamReceiver is responsible for the rtp-specific but
// media-independent state needed for receiving an RTP stream.
// TODO(bugs.webrtc.org/7135): Currently, only owns the association between ssrc
// and the stream's RtpPacketSinkInterface. Ownership of corresponding objects
// from modules/rtp_rtcp/ should move to this class (or rather, the
// corresponding implementation class). We should add methods for getting rtp
// receive stats, and for sending RTCP messages related to the receive stream.
class RtpStreamReceiverInterface {
 public:
  virtual ~RtpStreamReceiverInterface() {}
};

// This class acts as a factory for RtpStreamReceiver objects.
class RtpStreamReceiverControllerInterface {
 public:
  virtual ~RtpStreamReceiverControllerInterface() {}

  virtual std::unique_ptr<RtpStreamReceiverInterface> CreateReceiver(
      uint32_t ssrc,
      RtpPacketSinkInterface* sink) = 0;
};

}  // namespace webrtc

#endif  // CALL_RTP_STREAM_RECEIVER_CONTROLLER_INTERFACE_H_
