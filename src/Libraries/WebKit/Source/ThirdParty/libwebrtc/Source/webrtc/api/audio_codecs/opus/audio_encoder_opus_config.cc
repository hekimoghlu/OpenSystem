/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "api/audio_codecs/opus/audio_encoder_opus_config.h"

namespace webrtc {

namespace {

#if defined(WEBRTC_ANDROID) || defined(WEBRTC_IOS)
constexpr int kDefaultComplexity = 5;
#else
constexpr int kDefaultComplexity = 9;
#endif

constexpr int kDefaultLowRateComplexity =
    WEBRTC_OPUS_VARIABLE_COMPLEXITY ? 9 : kDefaultComplexity;

}  // namespace

constexpr int AudioEncoderOpusConfig::kDefaultFrameSizeMs;
constexpr int AudioEncoderOpusConfig::kMinBitrateBps;
constexpr int AudioEncoderOpusConfig::kMaxBitrateBps;

AudioEncoderOpusConfig::AudioEncoderOpusConfig()
    : frame_size_ms(kDefaultFrameSizeMs),
      sample_rate_hz(48000),
      num_channels(1),
      application(ApplicationMode::kVoip),
      bitrate_bps(32000),
      fec_enabled(false),
      cbr_enabled(false),
      max_playback_rate_hz(48000),
      complexity(kDefaultComplexity),
      low_rate_complexity(kDefaultLowRateComplexity),
      complexity_threshold_bps(12500),
      complexity_threshold_window_bps(1500),
      dtx_enabled(false),
      uplink_bandwidth_update_interval_ms(200),
      payload_type(-1) {}
AudioEncoderOpusConfig::AudioEncoderOpusConfig(const AudioEncoderOpusConfig&) =
    default;
AudioEncoderOpusConfig::~AudioEncoderOpusConfig() = default;
AudioEncoderOpusConfig& AudioEncoderOpusConfig::operator=(
    const AudioEncoderOpusConfig&) = default;

bool AudioEncoderOpusConfig::IsOk() const {
  if (frame_size_ms <= 0 || frame_size_ms % 10 != 0)
    return false;
  if (sample_rate_hz != 16000 && sample_rate_hz != 48000) {
    // Unsupported input sample rate. (libopus supports a few other rates as
    // well; we can add support for them when needed.)
    return false;
  }
  if (num_channels >= 255) {
    return false;
  }
  if (!bitrate_bps)
    return false;
  if (*bitrate_bps < kMinBitrateBps || *bitrate_bps > kMaxBitrateBps)
    return false;
  if (complexity < 0 || complexity > 10)
    return false;
  if (low_rate_complexity < 0 || low_rate_complexity > 10)
    return false;
  return true;
}
}  // namespace webrtc
