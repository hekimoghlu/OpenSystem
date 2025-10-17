/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#ifndef CALL_RTP_VIDEO_SENDER_INTERFACE_H_
#define CALL_RTP_VIDEO_SENDER_INTERFACE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "api/array_view.h"
#include "api/call/bitrate_allocation.h"
#include "api/fec_controller_override.h"
#include "api/video/video_layers_allocation.h"
#include "api/video_codecs/video_encoder.h"
#include "call/rtp_config.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_sequence_number_map.h"

namespace webrtc {
class VideoBitrateAllocation;
struct FecProtectionParams;

class RtpVideoSenderInterface : public EncodedImageCallback,
                                public FecControllerOverride {
 public:
  // Sets weather or not RTP packets is allowed to be sent on this sender.
  virtual void SetSending(bool enabled) = 0;
  virtual bool IsActive() = 0;

  virtual void OnNetworkAvailability(bool network_available) = 0;
  virtual std::map<uint32_t, RtpState> GetRtpStates() const = 0;
  virtual std::map<uint32_t, RtpPayloadState> GetRtpPayloadStates() const = 0;

  virtual void DeliverRtcp(const uint8_t* packet, size_t length) = 0;

  virtual void OnBitrateAllocationUpdated(
      const VideoBitrateAllocation& bitrate) = 0;
  virtual void OnVideoLayersAllocationUpdated(
      const VideoLayersAllocation& allocation) = 0;
  virtual void OnBitrateUpdated(BitrateAllocationUpdate update,
                                int framerate) = 0;
  virtual void OnTransportOverheadChanged(
      size_t transport_overhead_bytes_per_packet) = 0;
  virtual uint32_t GetPayloadBitrateBps() const = 0;
  virtual uint32_t GetProtectionBitrateBps() const = 0;
  virtual void SetEncodingData(size_t width,
                               size_t height,
                               size_t num_temporal_layers) = 0;
  virtual std::vector<RtpSequenceNumberMap::Info> GetSentRtpPacketInfos(
      uint32_t ssrc,
      rtc::ArrayView<const uint16_t> sequence_numbers) const = 0;

  // Implements FecControllerOverride.
  void SetFecAllowed(bool fec_allowed) override = 0;
};
}  // namespace webrtc
#endif  // CALL_RTP_VIDEO_SENDER_INTERFACE_H_
