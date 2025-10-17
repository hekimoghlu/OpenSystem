/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "common_audio/signal_processing/include/real_fft.h"

#include "common_audio/signal_processing/include/signal_processing_library.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

// FFT order.
const int kOrder = 5;
// Lengths for real FFT's time and frequency bufffers.
// For N-point FFT, the length requirements from API are N and N+2 respectively.
const int kTimeDataLength = 1 << kOrder;
const int kFreqDataLength = (1 << kOrder) + 2;
// For complex FFT's time and freq buffer. The implementation requires
// 2*N 16-bit words.
const int kComplexFftDataLength = 2 << kOrder;
// Reference data for time signal.
const int16_t kRefData[kTimeDataLength] = {
    11739,  6848,  -8688,  31980, -30295, 25242, 27085,  19410,
    -26299, 15607, -10791, 11778, -23819, 14498, -25772, 10076,
    1173,   6848,  -8688,  31980, -30295, 2522,  27085,  19410,
    -2629,  5607,  -3,     1178,  -23819, 1498,  -25772, 10076};

TEST(RealFFTTest, CreateFailsOnBadInput) {
  RealFFT* fft = WebRtcSpl_CreateRealFFT(11);
  EXPECT_TRUE(fft == nullptr);
  fft = WebRtcSpl_CreateRealFFT(-1);
  EXPECT_TRUE(fft == nullptr);
}

TEST(RealFFTTest, RealAndComplexMatch) {
  int i = 0;
  int j = 0;
  int16_t real_fft_time[kTimeDataLength] = {0};
  int16_t real_fft_freq[kFreqDataLength] = {0};
  // One common buffer for complex FFT's time and frequency data.
  int16_t complex_fft_buff[kComplexFftDataLength] = {0};

  // Prepare the inputs to forward FFT's.
  memcpy(real_fft_time, kRefData, sizeof(kRefData));
  for (i = 0, j = 0; i < kTimeDataLength; i += 1, j += 2) {
    complex_fft_buff[j] = kRefData[i];
    complex_fft_buff[j + 1] = 0;  // Insert zero's to imaginary parts.
  }

  // Create and run real forward FFT.
  RealFFT* fft = WebRtcSpl_CreateRealFFT(kOrder);
  EXPECT_TRUE(fft != nullptr);
  EXPECT_EQ(0, WebRtcSpl_RealForwardFFT(fft, real_fft_time, real_fft_freq));

  // Run complex forward FFT.
  WebRtcSpl_ComplexBitReverse(complex_fft_buff, kOrder);
  EXPECT_EQ(0, WebRtcSpl_ComplexFFT(complex_fft_buff, kOrder, 1));

  // Verify the results between complex and real forward FFT.
  for (i = 0; i < kFreqDataLength; i++) {
    EXPECT_EQ(real_fft_freq[i], complex_fft_buff[i]);
  }

  // Prepare the inputs to inverse real FFT.
  // We use whatever data in complex_fft_buff[] since we don't care
  // about data contents. Only kFreqDataLength 16-bit words are copied
  // from complex_fft_buff to real_fft_freq since remaining words (2nd half)
  // are conjugate-symmetric to the first half in theory.
  memcpy(real_fft_freq, complex_fft_buff, sizeof(real_fft_freq));

  // Run real inverse FFT.
  int real_scale = WebRtcSpl_RealInverseFFT(fft, real_fft_freq, real_fft_time);
  EXPECT_GE(real_scale, 0);

  // Run complex inverse FFT.
  WebRtcSpl_ComplexBitReverse(complex_fft_buff, kOrder);
  int complex_scale = WebRtcSpl_ComplexIFFT(complex_fft_buff, kOrder, 1);

  // Verify the results between complex and real inverse FFT.
  // They are not bit-exact, since complex IFFT doesn't produce
  // exactly conjugate-symmetric data (between first and second half).
  EXPECT_EQ(real_scale, complex_scale);
  for (i = 0, j = 0; i < kTimeDataLength; i += 1, j += 2) {
    EXPECT_LE(abs(real_fft_time[i] - complex_fft_buff[j]), 1);
  }

  WebRtcSpl_FreeRealFFT(fft);
}

}  // namespace
}  // namespace webrtc
