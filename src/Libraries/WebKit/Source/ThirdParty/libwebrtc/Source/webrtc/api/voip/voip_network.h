/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef API_VOIP_VOIP_NETWORK_H_
#define API_VOIP_VOIP_NETWORK_H_

#include <cstdint>

#include "api/array_view.h"
#include "api/voip/voip_base.h"

namespace webrtc {

// VoipNetwork interface provides any network related interfaces such as
// processing received RTP/RTCP packet from remote endpoint. This interface
// requires a ChannelId created via VoipBase interface.
class VoipNetwork {
 public:
  // The data received from the network including RTP header is passed here.
  // Returns following VoipResult;
  //  kOk - received RTP packet is processed.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult ReceivedRTPPacket(
      ChannelId channel_id,
      rtc::ArrayView<const uint8_t> rtp_packet) = 0;

  // The data received from the network including RTCP header is passed here.
  // Returns following VoipResult;
  //  kOk - received RTCP packet is processed.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult ReceivedRTCPPacket(
      ChannelId channel_id,
      rtc::ArrayView<const uint8_t> rtcp_packet) = 0;

 protected:
  virtual ~VoipNetwork() = default;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_NETWORK_H_
