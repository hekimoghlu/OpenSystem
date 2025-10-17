/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_H_

#include <optional>

#include "api/audio_codecs/audio_encoder.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor_config.h"

namespace webrtc {

// An AudioNetworkAdaptor optimizes the audio experience by suggesting a
// suitable runtime configuration (bit rate, frame length, FEC, etc.) to the
// encoder based on network metrics.
class AudioNetworkAdaptor {
 public:
  virtual ~AudioNetworkAdaptor() = default;

  virtual void SetUplinkBandwidth(int uplink_bandwidth_bps) = 0;

  virtual void SetUplinkPacketLossFraction(
      float uplink_packet_loss_fraction) = 0;

  virtual void SetRtt(int rtt_ms) = 0;

  virtual void SetTargetAudioBitrate(int target_audio_bitrate_bps) = 0;

  virtual void SetOverhead(size_t overhead_bytes_per_packet) = 0;

  virtual AudioEncoderRuntimeConfig GetEncoderRuntimeConfig() = 0;

  virtual void StartDebugDump(FILE* file_handle) = 0;

  virtual void StopDebugDump() = 0;

  virtual ANAStats GetStats() const = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_H_
