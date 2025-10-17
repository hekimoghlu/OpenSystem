/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#ifndef CALL_FLEXFEC_RECEIVE_STREAM_H_
#define CALL_FLEXFEC_RECEIVE_STREAM_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "api/call/transport.h"
#include "api/rtp_headers.h"
#include "call/receive_stream.h"
#include "call/rtp_packet_sink_interface.h"
#include "modules/rtp_rtcp/include/receive_statistics.h"

namespace webrtc {

class FlexfecReceiveStream : public RtpPacketSinkInterface,
                             public ReceiveStreamInterface {
 public:
  ~FlexfecReceiveStream() override = default;

  struct Config {
    explicit Config(Transport* rtcp_send_transport);
    Config(const Config&);
    ~Config();

    std::string ToString() const;

    // Returns true if all RTP information is available in order to
    // enable receiving FlexFEC.
    bool IsCompleteAndEnabled() const;

    // Payload type for FlexFEC.
    int payload_type = -1;

    ReceiveStreamRtpConfig rtp;

    // Vector containing a single element, corresponding to the SSRC of the
    // media stream being protected by this FlexFEC stream. The vector MUST have
    // size 1.
    //
    // TODO(brandtr): Update comment above when we support multistream
    // protection.
    std::vector<uint32_t> protected_media_ssrcs;

    // What RTCP mode to use in the reports.
    RtcpMode rtcp_mode = RtcpMode::kCompound;

    // Transport for outgoing RTCP packets.
    Transport* rtcp_send_transport = nullptr;
  };

  // TODO(tommi): FlexfecReceiveStream inherits from ReceiveStreamInterface,
  // not VideoReceiveStreamInterface where there's also a SetRtcpMode method.
  // Perhaps this should be in ReceiveStreamInterface and apply to audio streams
  // as well (although there's no logic that would use it at present).
  virtual void SetRtcpMode(RtcpMode mode) = 0;

  // Called to change the payload type after initialization.
  virtual void SetPayloadType(int payload_type) = 0;
  virtual int payload_type() const = 0;

  virtual const ReceiveStatistics* GetStats() const = 0;
};

}  // namespace webrtc

#endif  // CALL_FLEXFEC_RECEIVE_STREAM_H_
