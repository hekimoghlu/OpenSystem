/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include "modules/audio_processing/aec3/suppression_filter.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "test/gtest.h"

namespace webrtc {
namespace {

constexpr float kPi = 3.141592f;

void ProduceSinusoid(int sample_rate_hz,
                     float sinusoidal_frequency_hz,
                     size_t* sample_counter,
                     Block* x) {
  // Produce a sinusoid of the specified frequency.
  for (size_t k = *sample_counter, j = 0; k < (*sample_counter + kBlockSize);
       ++k, ++j) {
    for (int channel = 0; channel < x->NumChannels(); ++channel) {
      x->View(/*band=*/0, channel)[j] =
          32767.f *
          std::sin(2.f * kPi * sinusoidal_frequency_hz * k / sample_rate_hz);
    }
  }
  *sample_counter = *sample_counter + kBlockSize;

  for (int band = 1; band < x->NumBands(); ++band) {
    for (int channel = 0; channel < x->NumChannels(); ++channel) {
      std::fill(x->begin(band, channel), x->end(band, channel), 0.f);
    }
  }
}

}  // namespace

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

// Verifies the check for null suppressor output.
TEST(SuppressionFilterDeathTest, NullOutput) {
  std::vector<FftData> cn(1);
  std::vector<FftData> cn_high_bands(1);
  std::vector<FftData> E(1);
  std::array<float, kFftLengthBy2Plus1> gain;

  EXPECT_DEATH(SuppressionFilter(Aec3Optimization::kNone, 16000, 1)
                   .ApplyGain(cn, cn_high_bands, gain, 1.0f, E, nullptr),
               "");
}

// Verifies the check for allowed sample rate.
TEST(SuppressionFilterDeathTest, ProperSampleRate) {
  EXPECT_DEATH(SuppressionFilter(Aec3Optimization::kNone, 16001, 1), "");
}

#endif

// Verifies that no comfort noise is added when the gain is 1.
TEST(SuppressionFilter, ComfortNoiseInUnityGain) {
  SuppressionFilter filter(Aec3Optimization::kNone, 48000, 1);
  std::vector<FftData> cn(1);
  std::vector<FftData> cn_high_bands(1);
  std::array<float, kFftLengthBy2Plus1> gain;
  std::array<float, kFftLengthBy2> e_old_;
  Aec3Fft fft;

  e_old_.fill(0.f);
  gain.fill(1.f);
  cn[0].re.fill(1.f);
  cn[0].im.fill(1.f);
  cn_high_bands[0].re.fill(1.f);
  cn_high_bands[0].im.fill(1.f);

  Block e(3, kBlockSize);
  Block e_ref = e;

  std::vector<FftData> E(1);
  fft.PaddedFft(e.View(/*band=*/0, /*channel=*/0), e_old_,
                Aec3Fft::Window::kSqrtHanning, &E[0]);
  std::copy(e.begin(/*band=*/0, /*channel=*/0),
            e.end(/*band=*/0, /*channel=*/0), e_old_.begin());

  filter.ApplyGain(cn, cn_high_bands, gain, 1.f, E, &e);

  for (int band = 0; band < e.NumBands(); ++band) {
    for (int channel = 0; channel < e.NumChannels(); ++channel) {
      const auto e_view = e.View(band, channel);
      const auto e_ref_view = e_ref.View(band, channel);
      for (size_t sample = 0; sample < e_view.size(); ++sample) {
        EXPECT_EQ(e_ref_view[sample], e_view[sample]);
      }
    }
  }
}

// Verifies that the suppressor is able to suppress a signal.
TEST(SuppressionFilter, SignalSuppression) {
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);
  constexpr size_t kNumChannels = 1;

  SuppressionFilter filter(Aec3Optimization::kNone, kSampleRateHz, 1);
  std::vector<FftData> cn(1);
  std::vector<FftData> cn_high_bands(1);
  std::array<float, kFftLengthBy2> e_old_;
  Aec3Fft fft;
  std::array<float, kFftLengthBy2Plus1> gain;
  Block e(kNumBands, kNumChannels);
  e_old_.fill(0.f);

  gain.fill(1.f);
  std::for_each(gain.begin() + 10, gain.end(), [](float& a) { a = 0.f; });

  cn[0].re.fill(0.f);
  cn[0].im.fill(0.f);
  cn_high_bands[0].re.fill(0.f);
  cn_high_bands[0].im.fill(0.f);

  size_t sample_counter = 0;

  float e0_input = 0.f;
  float e0_output = 0.f;
  for (size_t k = 0; k < 100; ++k) {
    ProduceSinusoid(16000, 16000 * 40 / kFftLengthBy2 / 2, &sample_counter, &e);
    e0_input = std::inner_product(e.begin(/*band=*/0, /*channel=*/0),
                                  e.end(/*band=*/0, /*channel=*/0),
                                  e.begin(/*band=*/0, /*channel=*/0), e0_input);

    std::vector<FftData> E(1);
    fft.PaddedFft(e.View(/*band=*/0, /*channel=*/0), e_old_,
                  Aec3Fft::Window::kSqrtHanning, &E[0]);
    std::copy(e.begin(/*band=*/0, /*channel=*/0),
              e.end(/*band=*/0, /*channel=*/0), e_old_.begin());

    filter.ApplyGain(cn, cn_high_bands, gain, 1.f, E, &e);
    e0_output = std::inner_product(
        e.begin(/*band=*/0, /*channel=*/0), e.end(/*band=*/0, /*channel=*/0),
        e.begin(/*band=*/0, /*channel=*/0), e0_output);
  }

  EXPECT_LT(e0_output, e0_input / 1000.f);
}

// Verifies that the suppressor is able to pass through a desired signal while
// applying suppressing for some frequencies.
TEST(SuppressionFilter, SignalTransparency) {
  constexpr size_t kNumChannels = 1;
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);

  SuppressionFilter filter(Aec3Optimization::kNone, kSampleRateHz, 1);
  std::vector<FftData> cn(1);
  std::array<float, kFftLengthBy2> e_old_;
  Aec3Fft fft;
  std::vector<FftData> cn_high_bands(1);
  std::array<float, kFftLengthBy2Plus1> gain;
  Block e(kNumBands, kNumChannels);
  e_old_.fill(0.f);
  gain.fill(1.f);
  std::for_each(gain.begin() + 30, gain.end(), [](float& a) { a = 0.f; });

  cn[0].re.fill(0.f);
  cn[0].im.fill(0.f);
  cn_high_bands[0].re.fill(0.f);
  cn_high_bands[0].im.fill(0.f);

  size_t sample_counter = 0;

  float e0_input = 0.f;
  float e0_output = 0.f;
  for (size_t k = 0; k < 100; ++k) {
    ProduceSinusoid(16000, 16000 * 10 / kFftLengthBy2 / 2, &sample_counter, &e);
    e0_input = std::inner_product(e.begin(/*band=*/0, /*channel=*/0),
                                  e.end(/*band=*/0, /*channel=*/0),
                                  e.begin(/*band=*/0, /*channel=*/0), e0_input);

    std::vector<FftData> E(1);
    fft.PaddedFft(e.View(/*band=*/0, /*channel=*/0), e_old_,
                  Aec3Fft::Window::kSqrtHanning, &E[0]);
    std::copy(e.begin(/*band=*/0, /*channel=*/0),
              e.end(/*band=*/0, /*channel=*/0), e_old_.begin());

    filter.ApplyGain(cn, cn_high_bands, gain, 1.f, E, &e);
    e0_output = std::inner_product(
        e.begin(/*band=*/0, /*channel=*/0), e.end(/*band=*/0, /*channel=*/0),
        e.begin(/*band=*/0, /*channel=*/0), e0_output);
  }

  EXPECT_LT(0.9f * e0_input, e0_output);
}

// Verifies that the suppressor delay.
TEST(SuppressionFilter, Delay) {
  constexpr size_t kNumChannels = 1;
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);

  SuppressionFilter filter(Aec3Optimization::kNone, kSampleRateHz, 1);
  std::vector<FftData> cn(1);
  std::vector<FftData> cn_high_bands(1);
  std::array<float, kFftLengthBy2> e_old_;
  Aec3Fft fft;
  std::array<float, kFftLengthBy2Plus1> gain;
  Block e(kNumBands, kNumChannels);

  gain.fill(1.f);

  cn[0].re.fill(0.f);
  cn[0].im.fill(0.f);
  cn_high_bands[0].re.fill(0.f);
  cn_high_bands[0].im.fill(0.f);

  for (size_t k = 0; k < 100; ++k) {
    for (size_t band = 0; band < kNumBands; ++band) {
      for (size_t channel = 0; channel < kNumChannels; ++channel) {
        auto e_view = e.View(band, channel);
        for (size_t sample = 0; sample < kBlockSize; ++sample) {
          e_view[sample] = k * kBlockSize + sample + channel;
        }
      }
    }

    std::vector<FftData> E(1);
    fft.PaddedFft(e.View(/*band=*/0, /*channel=*/0), e_old_,
                  Aec3Fft::Window::kSqrtHanning, &E[0]);
    std::copy(e.begin(/*band=*/0, /*channel=*/0),
              e.end(/*band=*/0, /*channel=*/0), e_old_.begin());

    filter.ApplyGain(cn, cn_high_bands, gain, 1.f, E, &e);
    if (k > 2) {
      for (size_t band = 0; band < kNumBands; ++band) {
        for (size_t channel = 0; channel < kNumChannels; ++channel) {
          const auto e_view = e.View(band, channel);
          for (size_t sample = 0; sample < kBlockSize; ++sample) {
            EXPECT_NEAR(k * kBlockSize + sample - kBlockSize + channel,
                        e_view[sample], 0.01);
          }
        }
      }
    }
  }
}

}  // namespace webrtc
