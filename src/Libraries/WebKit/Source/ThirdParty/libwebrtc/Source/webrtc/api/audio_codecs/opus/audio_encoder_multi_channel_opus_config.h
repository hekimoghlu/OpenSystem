/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#ifndef API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_CONFIG_H_
#define API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_CONFIG_H_

#include <stddef.h>
#include <vector>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

struct RTC_EXPORT AudioEncoderMultiChannelOpusConfig {
  static constexpr int kDefaultFrameSizeMs = 20;

  // Opus API allows a min bitrate of 500bps, but Opus documentation suggests
  // bitrate should be in the range of 6000 to 510000, inclusive.
  static constexpr int kMinBitrateBps = 6000;
  static constexpr int kMaxBitrateBps = 510000;

  AudioEncoderMultiChannelOpusConfig();
  AudioEncoderMultiChannelOpusConfig(const AudioEncoderMultiChannelOpusConfig&);
  ~AudioEncoderMultiChannelOpusConfig();
  AudioEncoderMultiChannelOpusConfig& operator=(
      const AudioEncoderMultiChannelOpusConfig&);

  int frame_size_ms;
  size_t num_channels;
  enum class ApplicationMode { kVoip, kAudio };
  ApplicationMode application;
  int bitrate_bps;
  bool fec_enabled;
  bool cbr_enabled;
  bool dtx_enabled;
  int max_playback_rate_hz;
  std::vector<int> supported_frame_lengths_ms;

  int complexity;

  // Number of mono/stereo Opus streams.
  int num_streams;

  // Number of channel pairs coupled together, see RFC 7845 section
  // 5.1.1. Has to be less than the number of streams
  int coupled_streams;

  // Channel mapping table, defines the mapping from encoded streams to input
  // channels. See RFC 7845 section 5.1.1.
  std::vector<unsigned char> channel_mapping;

  bool IsOk() const;
};

}  // namespace webrtc
#endif  // API_AUDIO_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_CONFIG_H_
