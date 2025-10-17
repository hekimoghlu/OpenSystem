/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#ifndef COMMON_AUDIO_SMOOTHING_FILTER_H_
#define COMMON_AUDIO_SMOOTHING_FILTER_H_

#include <stdint.h>

#include <optional>

namespace webrtc {

class SmoothingFilter {
 public:
  virtual ~SmoothingFilter() = default;
  virtual void AddSample(float sample) = 0;
  virtual std::optional<float> GetAverage() = 0;
  virtual bool SetTimeConstantMs(int time_constant_ms) = 0;
};

// SmoothingFilterImpl applies an exponential filter
//   alpha = exp(-1.0 / time_constant_ms);
//   y[t] = alpha * y[t-1] + (1 - alpha) * sample;
// This implies a sample rate of 1000 Hz, i.e., 1 sample / ms.
// But SmoothingFilterImpl allows sparse samples. All missing samples will be
// assumed to equal the last received sample.
class SmoothingFilterImpl final : public SmoothingFilter {
 public:
  // `init_time_ms` is initialization time. It defines a period starting from
  // the arriving time of the first sample. During this period, the exponential
  // filter uses a varying time constant so that a smaller time constant will be
  // applied to the earlier samples. This is to allow the the filter to adapt to
  // earlier samples quickly. After the initialization period, the time constant
  // will be set to `init_time_ms` first and can be changed through
  // `SetTimeConstantMs`.
  explicit SmoothingFilterImpl(int init_time_ms);

  SmoothingFilterImpl() = delete;
  SmoothingFilterImpl(const SmoothingFilterImpl&) = delete;
  SmoothingFilterImpl& operator=(const SmoothingFilterImpl&) = delete;

  ~SmoothingFilterImpl() override;

  void AddSample(float sample) override;
  std::optional<float> GetAverage() override;
  bool SetTimeConstantMs(int time_constant_ms) override;

  // Methods used for unittests.
  float alpha() const { return alpha_; }

 private:
  void UpdateAlpha(int time_constant_ms);
  void ExtrapolateLastSample(int64_t time_ms);

  const int init_time_ms_;
  const float init_factor_;
  const float init_const_;

  std::optional<int64_t> init_end_time_ms_;
  float last_sample_;
  float alpha_;
  float state_;
  int64_t last_state_time_ms_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_SMOOTHING_FILTER_H_
