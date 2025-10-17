/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_BIQUAD_FILTER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_BIQUAD_FILTER_H_

#include "api/array_view.h"

namespace webrtc {

// Transposed direct form I implementation of a bi-quad filter.
//        b[0] + b[1] â€¢ z^(-1) + b[2] â€¢ z^(-2)
// H(z) = ------------------------------------
//          1 + a[1] â€¢ z^(-1) + a[2] â€¢ z^(-2)
class BiQuadFilter {
 public:
  // Normalized filter coefficients.
  // Computed as `[b, a] = scipy.signal.butter(N=2, Wn, btype)`.
  struct Config {
    float b[3];  // b[0], b[1], b[2].
    float a[2];  // a[1], a[2].
  };

  explicit BiQuadFilter(const Config& config);
  BiQuadFilter(const BiQuadFilter&) = delete;
  BiQuadFilter& operator=(const BiQuadFilter&) = delete;
  ~BiQuadFilter();

  // Sets the filter configuration and resets the internal state.
  void SetConfig(const Config& config);

  // Zeroes the filter state.
  void Reset();

  // Filters `x` and writes the output in `y`, which must have the same length
  // of `x`. In-place processing is supported.
  void Process(rtc::ArrayView<const float> x, rtc::ArrayView<float> y);

 private:
  Config config_;
  struct State {
    float b[2];
    float a[2];
  } state_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_BIQUAD_FILTER_H_
