/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "modules/audio_processing/agc2/vad_wrapper.h"

#include <array>
#include <utility>

#include "common_audio/resampler/include/push_resampler.h"
#include "modules/audio_processing/agc2/agc2_common.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/features_extraction.h"
#include "modules/audio_processing/agc2/rnn_vad/rnn.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

constexpr int kNumFramesPerSecond = 100;

class MonoVadImpl : public VoiceActivityDetectorWrapper::MonoVad {
 public:
  explicit MonoVadImpl(const AvailableCpuFeatures& cpu_features)
      : features_extractor_(cpu_features), rnn_vad_(cpu_features) {}
  MonoVadImpl(const MonoVadImpl&) = delete;
  MonoVadImpl& operator=(const MonoVadImpl&) = delete;
  ~MonoVadImpl() = default;

  int SampleRateHz() const override { return rnn_vad::kSampleRate24kHz; }
  void Reset() override { rnn_vad_.Reset(); }
  float Analyze(MonoView<const float> frame) override {
    RTC_DCHECK_EQ(frame.size(), rnn_vad::kFrameSize10ms24kHz);
    std::array<float, rnn_vad::kFeatureVectorSize> feature_vector;
    const bool is_silence = features_extractor_.CheckSilenceComputeFeatures(
        /*samples=*/{frame.data(), rnn_vad::kFrameSize10ms24kHz},
        feature_vector);
    return rnn_vad_.ComputeVadProbability(feature_vector, is_silence);
  }

 private:
  rnn_vad::FeaturesExtractor features_extractor_;
  rnn_vad::RnnVad rnn_vad_;
};

}  // namespace

VoiceActivityDetectorWrapper::VoiceActivityDetectorWrapper(
    const AvailableCpuFeatures& cpu_features,
    int sample_rate_hz)
    : VoiceActivityDetectorWrapper(kVadResetPeriodMs,
                                   cpu_features,
                                   sample_rate_hz) {}

VoiceActivityDetectorWrapper::VoiceActivityDetectorWrapper(
    int vad_reset_period_ms,
    const AvailableCpuFeatures& cpu_features,
    int sample_rate_hz)
    : VoiceActivityDetectorWrapper(vad_reset_period_ms,
                                   std::make_unique<MonoVadImpl>(cpu_features),
                                   sample_rate_hz) {}

VoiceActivityDetectorWrapper::VoiceActivityDetectorWrapper(
    int vad_reset_period_ms,
    std::unique_ptr<MonoVad> vad,
    int sample_rate_hz)
    : vad_reset_period_frames_(
          rtc::CheckedDivExact(vad_reset_period_ms, kFrameDurationMs)),
      frame_size_(rtc::CheckedDivExact(sample_rate_hz, kNumFramesPerSecond)),
      time_to_vad_reset_(vad_reset_period_frames_),
      vad_(std::move(vad)),
      resampled_buffer_(
          rtc::CheckedDivExact(vad_->SampleRateHz(), kNumFramesPerSecond)),
      resampler_(frame_size_,
                 resampled_buffer_.size(),
                 /*num_channels=*/1) {
  RTC_DCHECK_GT(vad_reset_period_frames_, 1);
  vad_->Reset();
}

VoiceActivityDetectorWrapper::~VoiceActivityDetectorWrapper() = default;

float VoiceActivityDetectorWrapper::Analyze(
    DeinterleavedView<const float> frame) {
  // Periodically reset the VAD.
  time_to_vad_reset_--;
  if (time_to_vad_reset_ <= 0) {
    vad_->Reset();
    time_to_vad_reset_ = vad_reset_period_frames_;
  }

  // Resample the first channel of `frame`.
  RTC_DCHECK_EQ(frame.samples_per_channel(), frame_size_);
  MonoView<float> dst(resampled_buffer_.data(), resampled_buffer_.size());
  resampler_.Resample(frame[0], dst);

  return vad_->Analyze(resampled_buffer_);
}

}  // namespace webrtc
