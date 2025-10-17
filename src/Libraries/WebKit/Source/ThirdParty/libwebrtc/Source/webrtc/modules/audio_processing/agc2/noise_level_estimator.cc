/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "modules/audio_processing/agc2/noise_level_estimator.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "api/audio/audio_view.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

constexpr int kFramesPerSecond = 100;

float FrameEnergy(DeinterleavedView<const float> audio) {
  float energy = 0.0f;
  for (size_t k = 0; k < audio.num_channels(); ++k) {
    MonoView<const float> ch = audio[k];
    float channel_energy =
        std::accumulate(ch.begin(), ch.end(), 0.0f,
                        [](float a, float b) -> float { return a + b * b; });
    energy = std::max(channel_energy, energy);
  }
  return energy;
}

float EnergyToDbfs(float signal_energy, int num_samples) {
  RTC_DCHECK_GE(signal_energy, 0.0f);
  const float rms_square = signal_energy / num_samples;
  constexpr float kMinDbfs = -90.30899869919436f;
  if (rms_square <= 1.0f) {
    return kMinDbfs;
  }
  return 10.0f * std::log10(rms_square) + kMinDbfs;
}

// Updates the noise floor with instant decay and slow attack. This tuning is
// specific for AGC2, so that (i) it can promptly increase the gain if the noise
// floor drops (instant decay) and (ii) in case of music or fast speech, due to
// which the noise floor can be overestimated, the gain reduction is slowed
// down.
float SmoothNoiseFloorEstimate(float current_estimate, float new_estimate) {
  constexpr float kAttack = 0.5f;
  if (current_estimate < new_estimate) {
    // Attack phase.
    return kAttack * new_estimate + (1.0f - kAttack) * current_estimate;
  }
  // Instant attack.
  return new_estimate;
}

class NoiseFloorEstimator : public NoiseLevelEstimator {
 public:
  // Update the noise floor every 5 seconds.
  static constexpr int kUpdatePeriodNumFrames = 500;
  static_assert(kUpdatePeriodNumFrames >= 200,
                "A too small value may cause noise level overestimation.");
  static_assert(kUpdatePeriodNumFrames <= 1500,
                "A too large value may make AGC2 slow at reacting to increased "
                "noise levels.");

  NoiseFloorEstimator(ApmDataDumper* data_dumper) : data_dumper_(data_dumper) {
    RTC_DCHECK(data_dumper_);
    // Initially assume that 48 kHz will be used. `Analyze()` will detect the
    // used sample rate and call `Initialize()` again if needed.
    Initialize(/*sample_rate_hz=*/48000);
  }
  NoiseFloorEstimator(const NoiseFloorEstimator&) = delete;
  NoiseFloorEstimator& operator=(const NoiseFloorEstimator&) = delete;
  ~NoiseFloorEstimator() = default;

  float Analyze(DeinterleavedView<const float> frame) override {
    // Detect sample rate changes.
    const int sample_rate_hz =
        static_cast<int>(frame.samples_per_channel() * kFramesPerSecond);
    if (sample_rate_hz != sample_rate_hz_) {
      Initialize(sample_rate_hz);
    }

    const float frame_energy = FrameEnergy(frame);
    if (frame_energy <= min_noise_energy_) {
      // Ignore frames when muted or below the minimum measurable energy.
      if (data_dumper_)
        data_dumper_->DumpRaw("agc2_noise_floor_estimator_preliminary_level",
                              noise_energy_);
      return EnergyToDbfs(noise_energy_,
                          static_cast<int>(frame.samples_per_channel()));
    }

    if (preliminary_noise_energy_set_) {
      preliminary_noise_energy_ =
          std::min(preliminary_noise_energy_, frame_energy);
    } else {
      preliminary_noise_energy_ = frame_energy;
      preliminary_noise_energy_set_ = true;
    }
    if (data_dumper_)
      data_dumper_->DumpRaw("agc2_noise_floor_estimator_preliminary_level",
                            preliminary_noise_energy_);

    if (counter_ == 0) {
      // Full period observed.
      first_period_ = false;
      // Update the estimated noise floor energy with the preliminary
      // estimation.
      noise_energy_ = SmoothNoiseFloorEstimate(
          /*current_estimate=*/noise_energy_,
          /*new_estimate=*/preliminary_noise_energy_);
      // Reset for a new observation period.
      counter_ = kUpdatePeriodNumFrames;
      preliminary_noise_energy_set_ = false;
    } else if (first_period_) {
      // While analyzing the signal during the initial period, continuously
      // update the estimated noise energy, which is monotonic.
      noise_energy_ = preliminary_noise_energy_;
      counter_--;
    } else {
      // During the observation period it's only allowed to lower the energy.
      noise_energy_ = std::min(noise_energy_, preliminary_noise_energy_);
      counter_--;
    }

    float noise_rms_dbfs = EnergyToDbfs(
        noise_energy_, static_cast<int>(frame.samples_per_channel()));
    if (data_dumper_)
      data_dumper_->DumpRaw("agc2_noise_rms_dbfs", noise_rms_dbfs);

    return noise_rms_dbfs;
  }

 private:
  void Initialize(int sample_rate_hz) {
    sample_rate_hz_ = sample_rate_hz;
    first_period_ = true;
    preliminary_noise_energy_set_ = false;
    // Initialize the minimum noise energy to -84 dBFS.
    min_noise_energy_ = sample_rate_hz * 2.0f * 2.0f / kFramesPerSecond;
    preliminary_noise_energy_ = min_noise_energy_;
    noise_energy_ = min_noise_energy_;
    counter_ = kUpdatePeriodNumFrames;
  }

  ApmDataDumper* const data_dumper_;
  int sample_rate_hz_;
  float min_noise_energy_;
  bool first_period_;
  bool preliminary_noise_energy_set_;
  float preliminary_noise_energy_;
  float noise_energy_;
  int counter_;
};

}  // namespace

std::unique_ptr<NoiseLevelEstimator> CreateNoiseFloorEstimator(
    ApmDataDumper* data_dumper) {
  return std::make_unique<NoiseFloorEstimator>(data_dumper);
}

}  // namespace webrtc
