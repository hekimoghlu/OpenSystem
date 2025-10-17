/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_VIDEO_FEC_GENERATOR_H_
#define MODULES_RTP_RTCP_SOURCE_VIDEO_FEC_GENERATOR_H_

#include <memory>
#include <vector>

#include "api/units/data_rate.h"
#include "modules/include/module_fec_types.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"

namespace webrtc {

class VideoFecGenerator {
 public:
  VideoFecGenerator() = default;
  virtual ~VideoFecGenerator() = default;

  enum class FecType { kFlexFec, kUlpFec };
  virtual FecType GetFecType() const = 0;
  // Returns the SSRC used for FEC packets (i.e. FlexFec SSRC).
  virtual std::optional<uint32_t> FecSsrc() = 0;
  // Returns the overhead, in bytes per packet, for FEC (and possibly RED).
  virtual size_t MaxPacketOverhead() const = 0;
  // Current rate of FEC packets generated, including all RTP-level headers.
  virtual DataRate CurrentFecRate() const = 0;
  // Set FEC rates, max frames before FEC is sent, and type of FEC masks.
  virtual void SetProtectionParameters(
      const FecProtectionParams& delta_params,
      const FecProtectionParams& key_params) = 0;
  // Called on new media packet to be protected. The generator may choose
  // to generate FEC packets at this time, if so they will be stored in an
  // internal buffer.
  virtual void AddPacketAndGenerateFec(const RtpPacketToSend& packet) = 0;
  // Get (and remove) and FEC packets pending in the generator. These packets
  // will lack sequence numbers, that needs to be set externally.
  // TODO(bugs.webrtc.org/11340): Actually FlexFec sets seq#, fix that!
  virtual std::vector<std::unique_ptr<RtpPacketToSend>> GetFecPackets() = 0;
  // Only called on the VideoSendStream queue, after operation has shut down,
  // and only populated if there is an RtpState (e.g. FlexFec).
  virtual std::optional<RtpState> GetRtpState() = 0;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_VIDEO_FEC_GENERATOR_H_
