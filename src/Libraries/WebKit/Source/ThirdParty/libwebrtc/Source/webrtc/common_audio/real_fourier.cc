/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "common_audio/real_fourier.h"

#include "common_audio/real_fourier_ooura.h"
#include "common_audio/signal_processing/include/signal_processing_library.h"
#include "rtc_base/checks.h"

namespace webrtc {

using std::complex;

const size_t RealFourier::kFftBufferAlignment = 32;

std::unique_ptr<RealFourier> RealFourier::Create(int fft_order) {
  return std::unique_ptr<RealFourier>(new RealFourierOoura(fft_order));
}

int RealFourier::FftOrder(size_t length) {
  RTC_CHECK_GT(length, 0U);
  return WebRtcSpl_GetSizeInBits(static_cast<uint32_t>(length - 1));
}

size_t RealFourier::FftLength(int order) {
  RTC_CHECK_GE(order, 0);
  return size_t{1} << order;
}

size_t RealFourier::ComplexLength(int order) {
  return FftLength(order) / 2 + 1;
}

RealFourier::fft_real_scoper RealFourier::AllocRealBuffer(int count) {
  return fft_real_scoper(static_cast<float*>(
      AlignedMalloc(sizeof(float) * count, kFftBufferAlignment)));
}

RealFourier::fft_cplx_scoper RealFourier::AllocCplxBuffer(int count) {
  return fft_cplx_scoper(static_cast<complex<float>*>(
      AlignedMalloc(sizeof(complex<float>) * count, kFftBufferAlignment)));
}

}  // namespace webrtc
