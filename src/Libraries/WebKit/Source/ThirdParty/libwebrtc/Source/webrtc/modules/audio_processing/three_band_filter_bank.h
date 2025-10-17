/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_
#define MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "api/array_view.h"

namespace webrtc {

constexpr int kSparsity = 4;
constexpr int kStrideLog2 = 2;
constexpr int kStride = 1 << kStrideLog2;
constexpr int kNumZeroFilters = 2;
constexpr int kFilterSize = 4;
constexpr int kMemorySize = kFilterSize * kStride - 1;
static_assert(kMemorySize == 15,
              "The memory size must be sufficient to provide memory for the "
              "shifted filters");

// An implementation of a 3-band FIR filter-bank with DCT modulation, similar to
// the proposed in "Multirate Signal Processing for Communication Systems" by
// Fredric J Harris.
// The low-pass filter prototype has these characteristics:
// * Pass-band ripple = 0.3dB
// * Pass-band frequency = 0.147 (7kHz at 48kHz)
// * Stop-band attenuation = 40dB
// * Stop-band frequency = 0.192 (9.2kHz at 48kHz)
// * Delay = 24 samples (500us at 48kHz)
// * Linear phase
// This filter bank does not satisfy perfect reconstruction. The SNR after
// analysis and synthesis (with no processing in between) is approximately 9.5dB
// depending on the input signal after compensating for the delay.
class ThreeBandFilterBank final {
 public:
  static const int kNumBands = 3;
  static const int kFullBandSize = 480;
  static const int kSplitBandSize =
      ThreeBandFilterBank::kFullBandSize / ThreeBandFilterBank::kNumBands;
  static const int kNumNonZeroFilters =
      kSparsity * ThreeBandFilterBank::kNumBands - kNumZeroFilters;

  ThreeBandFilterBank();
  ~ThreeBandFilterBank();

  // Splits `in` of size kFullBandSize into 3 downsampled frequency bands in
  // `out`, each of size 160.
  void Analysis(rtc::ArrayView<const float, kFullBandSize> in,
                rtc::ArrayView<const rtc::ArrayView<float>, kNumBands> out);

  // Merges the 3 downsampled frequency bands in `in`, each of size 160, into
  // `out`, which is of size kFullBandSize.
  void Synthesis(rtc::ArrayView<const rtc::ArrayView<float>, kNumBands> in,
                 rtc::ArrayView<float, kFullBandSize> out);

 private:
  std::array<std::array<float, kMemorySize>, kNumNonZeroFilters>
      state_analysis_;
  std::array<std::array<float, kMemorySize>, kNumNonZeroFilters>
      state_synthesis_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_THREE_BAND_FILTER_BANK_H_
