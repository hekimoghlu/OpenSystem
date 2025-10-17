/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#include "modules/audio_processing/agc2/gain_applier.h"

#include "api/audio/audio_view.h"
#include "modules/audio_processing/agc2/agc2_common.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {
namespace {

// Returns true when the gain factor is so close to 1 that it would
// not affect int16 samples.
bool GainCloseToOne(float gain_factor) {
  return 1.f - 1.f / kMaxFloatS16Value <= gain_factor &&
         gain_factor <= 1.f + 1.f / kMaxFloatS16Value;
}

void ClipSignal(DeinterleavedView<float> signal) {
  for (size_t k = 0; k < signal.num_channels(); ++k) {
    MonoView<float> channel_view = signal[k];
    for (auto& sample : channel_view) {
      sample = rtc::SafeClamp(sample, kMinFloatS16Value, kMaxFloatS16Value);
    }
  }
}

void ApplyGainWithRamping(float last_gain_linear,
                          float gain_at_end_of_frame_linear,
                          float inverse_samples_per_channel,
                          DeinterleavedView<float> float_frame) {
  // Do not modify the signal.
  if (last_gain_linear == gain_at_end_of_frame_linear &&
      GainCloseToOne(gain_at_end_of_frame_linear)) {
    return;
  }

  // Gain is constant and different from 1.
  if (last_gain_linear == gain_at_end_of_frame_linear) {
    for (size_t k = 0; k < float_frame.num_channels(); ++k) {
      MonoView<float> channel_view = float_frame[k];
      for (auto& sample : channel_view) {
        sample *= gain_at_end_of_frame_linear;
      }
    }
    return;
  }

  // The gain changes. We have to change slowly to avoid discontinuities.
  const float increment = (gain_at_end_of_frame_linear - last_gain_linear) *
                          inverse_samples_per_channel;
  for (size_t ch = 0; ch < float_frame.num_channels(); ++ch) {
    float gain = last_gain_linear;
    for (float& sample : float_frame[ch]) {
      sample *= gain;
      gain += increment;
    }
  }
}

}  // namespace

GainApplier::GainApplier(bool hard_clip_samples, float initial_gain_factor)
    : hard_clip_samples_(hard_clip_samples),
      last_gain_factor_(initial_gain_factor),
      current_gain_factor_(initial_gain_factor) {}

void GainApplier::ApplyGain(DeinterleavedView<float> signal) {
  if (static_cast<int>(signal.samples_per_channel()) != samples_per_channel_) {
    Initialize(signal.samples_per_channel());
  }

  ApplyGainWithRamping(last_gain_factor_, current_gain_factor_,
                       inverse_samples_per_channel_, signal);

  last_gain_factor_ = current_gain_factor_;

  if (hard_clip_samples_) {
    ClipSignal(signal);
  }
}

// TODO(bugs.webrtc.org/7494): Remove once switched to gains in dB.
void GainApplier::SetGainFactor(float gain_factor) {
  RTC_DCHECK_GT(gain_factor, 0.f);
  current_gain_factor_ = gain_factor;
}

void GainApplier::Initialize(int samples_per_channel) {
  RTC_DCHECK_GT(samples_per_channel, 0);
  samples_per_channel_ = static_cast<int>(samples_per_channel);
  inverse_samples_per_channel_ = 1.f / samples_per_channel_;
}

}  // namespace webrtc
