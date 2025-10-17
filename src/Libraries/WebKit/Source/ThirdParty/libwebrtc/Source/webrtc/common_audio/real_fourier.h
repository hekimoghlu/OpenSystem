/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#ifndef COMMON_AUDIO_REAL_FOURIER_H_
#define COMMON_AUDIO_REAL_FOURIER_H_

#include <stddef.h>

#include <complex>
#include <memory>

#include "rtc_base/memory/aligned_malloc.h"

// Uniform interface class for the real DFT and its inverse, for power-of-2
// input lengths. Also contains helper functions for buffer allocation, taking
// care of any memory alignment requirements the underlying library might have.

namespace webrtc {

class RealFourier {
 public:
  // Shorthand typenames for the scopers used by the buffer allocation helpers.
  typedef std::unique_ptr<float[], AlignedFreeDeleter> fft_real_scoper;
  typedef std::unique_ptr<std::complex<float>[], AlignedFreeDeleter>
      fft_cplx_scoper;

  // The alignment required for all input and output buffers, in bytes.
  static const size_t kFftBufferAlignment;

  // Construct a wrapper instance for the given input order, which must be
  // between 1 and kMaxFftOrder, inclusively.
  static std::unique_ptr<RealFourier> Create(int fft_order);
  virtual ~RealFourier() {}

  // Helper to compute the smallest FFT order (a power of 2) which will contain
  // the given input length.
  static int FftOrder(size_t length);

  // Helper to compute the input length from the FFT order.
  static size_t FftLength(int order);

  // Helper to compute the exact length, in complex floats, of the transform
  // output (i.e. |2^order / 2 + 1|).
  static size_t ComplexLength(int order);

  // Buffer allocation helpers. The buffers are large enough to hold `count`
  // floats/complexes and suitably aligned for use by the implementation.
  // The returned scopers are set up with proper deleters; the caller owns
  // the allocated memory.
  static fft_real_scoper AllocRealBuffer(int count);
  static fft_cplx_scoper AllocCplxBuffer(int count);

  // Main forward transform interface. The output array need only be big
  // enough for |2^order / 2 + 1| elements - the conjugate pairs are not
  // returned. Input and output must be properly aligned (e.g. through
  // AllocRealBuffer and AllocCplxBuffer) and input length must be
  // |2^order| (same as given at construction time).
  virtual void Forward(const float* src, std::complex<float>* dest) const = 0;

  // Inverse transform. Same input format as output above, conjugate pairs
  // not needed.
  virtual void Inverse(const std::complex<float>* src, float* dest) const = 0;

  virtual int order() const = 0;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_REAL_FOURIER_H_
