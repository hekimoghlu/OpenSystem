/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "modules/audio_processing/aec3/decimator.h"

#include <array>
#include <vector>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

// signal.butter(2, 3400/8000.0, 'lowpass', analog=False)
const std::vector<CascadedBiQuadFilter::BiQuadParam> GetLowPassFilterDS2() {
  return std::vector<CascadedBiQuadFilter::BiQuadParam>{
      {{-1.f, 0.f}, {0.13833231f, 0.40743176f}, 0.22711796393486466f},
      {{-1.f, 0.f}, {0.13833231f, 0.40743176f}, 0.22711796393486466f},
      {{-1.f, 0.f}, {0.13833231f, 0.40743176f}, 0.22711796393486466f}};
}

// signal.ellip(6, 1, 40, 1800/8000, btype='lowpass', analog=False)
const std::vector<CascadedBiQuadFilter::BiQuadParam> GetLowPassFilterDS4() {
  return std::vector<CascadedBiQuadFilter::BiQuadParam>{
      {{-0.08873842f, 0.99605496f}, {0.75916227f, 0.23841065f}, 0.26250696827f},
      {{0.62273832f, 0.78243018f}, {0.74892112f, 0.5410152f}, 0.26250696827f},
      {{0.71107693f, 0.70311421f}, {0.74895534f, 0.63924616f}, 0.26250696827f}};
}

// signal.cheby1(1, 6, [1000/8000, 2000/8000], btype='bandpass', analog=False)
const std::vector<CascadedBiQuadFilter::BiQuadParam> GetBandPassFilterDS8() {
  return std::vector<CascadedBiQuadFilter::BiQuadParam>{
      {{1.f, 0.f}, {0.7601815f, 0.46423542f}, 0.10330478266505948f, true},
      {{1.f, 0.f}, {0.7601815f, 0.46423542f}, 0.10330478266505948f, true},
      {{1.f, 0.f}, {0.7601815f, 0.46423542f}, 0.10330478266505948f, true},
      {{1.f, 0.f}, {0.7601815f, 0.46423542f}, 0.10330478266505948f, true},
      {{1.f, 0.f}, {0.7601815f, 0.46423542f}, 0.10330478266505948f, true}};
}

// signal.butter(2, 1000/8000.0, 'highpass', analog=False)
const std::vector<CascadedBiQuadFilter::BiQuadParam> GetHighPassFilter() {
  return std::vector<CascadedBiQuadFilter::BiQuadParam>{
      {{1.f, 0.f}, {0.72712179f, 0.21296904f}, 0.7570763753338849f}};
}

const std::vector<CascadedBiQuadFilter::BiQuadParam> GetPassThroughFilter() {
  return std::vector<CascadedBiQuadFilter::BiQuadParam>{};
}
}  // namespace

Decimator::Decimator(size_t down_sampling_factor)
    : down_sampling_factor_(down_sampling_factor),
      anti_aliasing_filter_(down_sampling_factor_ == 4
                                ? GetLowPassFilterDS4()
                                : (down_sampling_factor_ == 8
                                       ? GetBandPassFilterDS8()
                                       : GetLowPassFilterDS2())),
      noise_reduction_filter_(down_sampling_factor_ == 8
                                  ? GetPassThroughFilter()
                                  : GetHighPassFilter()) {
  RTC_DCHECK(down_sampling_factor_ == 2 || down_sampling_factor_ == 4 ||
             down_sampling_factor_ == 8);
}

void Decimator::Decimate(rtc::ArrayView<const float> in,
                         rtc::ArrayView<float> out) {
  RTC_DCHECK_EQ(kBlockSize, in.size());
  RTC_DCHECK_EQ(kBlockSize / down_sampling_factor_, out.size());
  std::array<float, kBlockSize> x;

  // Limit the frequency content of the signal to avoid aliasing.
  anti_aliasing_filter_.Process(in, x);

  // Reduce the impact of near-end noise.
  noise_reduction_filter_.Process(x);

  // Downsample the signal.
  for (size_t j = 0, k = 0; j < out.size(); ++j, k += down_sampling_factor_) {
    RTC_DCHECK_GT(kBlockSize, k);
    out[j] = x[k];
  }
}

}  // namespace webrtc
