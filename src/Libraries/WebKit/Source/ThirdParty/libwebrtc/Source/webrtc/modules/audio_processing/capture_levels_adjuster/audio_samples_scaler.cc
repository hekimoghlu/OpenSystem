/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "modules/audio_processing/capture_levels_adjuster/audio_samples_scaler.h"

#include <algorithm>

#include "api/array_view.h"
#include "modules/audio_processing/audio_buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {

AudioSamplesScaler::AudioSamplesScaler(float initial_gain)
    : previous_gain_(initial_gain), target_gain_(initial_gain) {}

void AudioSamplesScaler::Process(AudioBuffer& audio_buffer) {
  if (static_cast<int>(audio_buffer.num_frames()) != samples_per_channel_) {
    // Update the members depending on audio-buffer length if needed.
    RTC_DCHECK_GT(audio_buffer.num_frames(), 0);
    samples_per_channel_ = static_cast<int>(audio_buffer.num_frames());
    one_by_samples_per_channel_ = 1.f / samples_per_channel_;
  }

  if (target_gain_ == 1.f && previous_gain_ == target_gain_) {
    // If only a gain of 1 is to be applied, do an early return without applying
    // any gain.
    return;
  }

  float gain = previous_gain_;
  if (previous_gain_ == target_gain_) {
    // Apply a non-changing gain.
    for (size_t channel = 0; channel < audio_buffer.num_channels(); ++channel) {
      rtc::ArrayView<float> channel_view(audio_buffer.channels()[channel],
                                         samples_per_channel_);
      for (float& sample : channel_view) {
        sample *= gain;
      }
    }
  } else {
    const float increment =
        (target_gain_ - previous_gain_) * one_by_samples_per_channel_;

    if (increment > 0.f) {
      // Apply an increasing gain.
      for (size_t channel = 0; channel < audio_buffer.num_channels();
           ++channel) {
        gain = previous_gain_;
        rtc::ArrayView<float> channel_view(audio_buffer.channels()[channel],
                                           samples_per_channel_);
        for (float& sample : channel_view) {
          gain = std::min(gain + increment, target_gain_);
          sample *= gain;
        }
      }
    } else {
      // Apply a decreasing gain.
      for (size_t channel = 0; channel < audio_buffer.num_channels();
           ++channel) {
        gain = previous_gain_;
        rtc::ArrayView<float> channel_view(audio_buffer.channels()[channel],
                                           samples_per_channel_);
        for (float& sample : channel_view) {
          gain = std::max(gain + increment, target_gain_);
          sample *= gain;
        }
      }
    }
  }
  previous_gain_ = target_gain_;

  // Saturate the samples to be in the S16 range.
  for (size_t channel = 0; channel < audio_buffer.num_channels(); ++channel) {
    rtc::ArrayView<float> channel_view(audio_buffer.channels()[channel],
                                       samples_per_channel_);
    for (float& sample : channel_view) {
      constexpr float kMinFloatS16Value = -32768.f;
      constexpr float kMaxFloatS16Value = 32767.f;
      sample = rtc::SafeClamp(sample, kMinFloatS16Value, kMaxFloatS16Value);
    }
  }
}

}  // namespace webrtc
