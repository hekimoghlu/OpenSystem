/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_COMMON_H_
#define DALI_KERNELS_SIGNAL_FFT_FFT_COMMON_H_

#include <complex>

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

enum FftSpectrumType {
  FFT_SPECTRUM_COMPLEX = 0,    // separate interleaved real and imag parts: (r0, i0, r1, i1, ...)
  FFT_SPECTRUM_MAGNITUDE = 1,  // sqrt( real^2 + imag^2 )
  FFT_SPECTRUM_POWER = 2,      // real^2 + imag^2
  FFT_SPECTRUM_POWER_DECIBELS = 3,  // 10 * log10(real^2 + imag^2)
};

using complexf = std::complex<float>;

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_COMMON_H_
