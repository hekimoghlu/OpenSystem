/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_CONFIG_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_CONFIG_H_

#include <stddef.h>

#include <optional>

namespace webrtc {

struct AudioEncoderRuntimeConfig {
  AudioEncoderRuntimeConfig();
  AudioEncoderRuntimeConfig(const AudioEncoderRuntimeConfig& other);
  ~AudioEncoderRuntimeConfig();
  AudioEncoderRuntimeConfig& operator=(const AudioEncoderRuntimeConfig& other);
  bool operator==(const AudioEncoderRuntimeConfig& other) const;
  std::optional<int> bitrate_bps;
  std::optional<int> frame_length_ms;
  // Note: This is what we tell the encoder. It doesn't have to reflect
  // the actual NetworkMetrics; it's subject to our decision.
  std::optional<float> uplink_packet_loss_fraction;
  std::optional<bool> enable_fec;
  std::optional<bool> enable_dtx;

  // Some encoders can encode fewer channels than the actual input to make
  // better use of the bandwidth. `num_channels` sets the number of channels
  // to encode.
  std::optional<size_t> num_channels;

  // This is true if the last frame length change was an increase, and otherwise
  // false.
  // The value of this boolean is used to apply a different offset to the
  // per-packet overhead that is reported by the BWE. The exact offset value
  // is most important right after a frame length change, because the frame
  // length change affects the overhead. In the steady state, the exact value is
  // not important because the BWE will compensate.
  bool last_fl_change_increase = false;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_INCLUDE_AUDIO_NETWORK_ADAPTOR_CONFIG_H_
