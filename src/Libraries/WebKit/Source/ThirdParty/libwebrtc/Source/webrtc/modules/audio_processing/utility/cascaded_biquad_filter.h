/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_UTILITY_CASCADED_BIQUAD_FILTER_H_
#define MODULES_AUDIO_PROCESSING_UTILITY_CASCADED_BIQUAD_FILTER_H_

#include <stddef.h>

#include <complex>
#include <vector>

#include "api/array_view.h"

namespace webrtc {

// Applies a number of biquads in a cascaded manner. The filter implementation
// is direct form 1.
class CascadedBiQuadFilter {
 public:
  struct BiQuadParam {
    BiQuadParam(std::complex<float> zero,
                std::complex<float> pole,
                float gain,
                bool mirror_zero_along_i_axis = false);
    explicit BiQuadParam(const BiQuadParam&);
    std::complex<float> zero;
    std::complex<float> pole;
    float gain;
    bool mirror_zero_along_i_axis;
  };

  struct BiQuadCoefficients {
    float b[3];
    float a[2];
  };

  struct BiQuad {
    explicit BiQuad(const BiQuadCoefficients& coefficients)
        : coefficients(coefficients), x(), y() {}
    explicit BiQuad(const CascadedBiQuadFilter::BiQuadParam& param);
    void Reset();
    BiQuadCoefficients coefficients;
    float x[2];
    float y[2];
  };

  CascadedBiQuadFilter(
      const CascadedBiQuadFilter::BiQuadCoefficients& coefficients,
      size_t num_biquads);
  explicit CascadedBiQuadFilter(
      const std::vector<CascadedBiQuadFilter::BiQuadParam>& biquad_params);
  ~CascadedBiQuadFilter();
  CascadedBiQuadFilter(const CascadedBiQuadFilter&) = delete;
  CascadedBiQuadFilter& operator=(const CascadedBiQuadFilter&) = delete;

  // Applies the biquads on the values in x in order to form the output in y.
  void Process(rtc::ArrayView<const float> x, rtc::ArrayView<float> y);
  // Applies the biquads on the values in y in an in-place manner.
  void Process(rtc::ArrayView<float> y);
  // Resets the filter to its initial state.
  void Reset();

 private:
  void ApplyBiQuad(rtc::ArrayView<const float> x,
                   rtc::ArrayView<float> y,
                   CascadedBiQuadFilter::BiQuad* biquad);

  std::vector<BiQuad> biquads_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_UTILITY_CASCADED_BIQUAD_FILTER_H_
