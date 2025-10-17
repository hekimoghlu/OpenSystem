/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "modules/audio_processing/aec3/aec3_fft.h"

#include <algorithm>

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

// Verifies that the check for non-null input in Fft works.
TEST(Aec3FftDeathTest, NullFftInput) {
  Aec3Fft fft;
  FftData X;
  EXPECT_DEATH(fft.Fft(nullptr, &X), "");
}

// Verifies that the check for non-null input in Fft works.
TEST(Aec3FftDeathTest, NullFftOutput) {
  Aec3Fft fft;
  std::array<float, kFftLength> x;
  EXPECT_DEATH(fft.Fft(&x, nullptr), "");
}

// Verifies that the check for non-null output in Ifft works.
TEST(Aec3FftDeathTest, NullIfftOutput) {
  Aec3Fft fft;
  FftData X;
  EXPECT_DEATH(fft.Ifft(X, nullptr), "");
}

// Verifies that the check for non-null output in ZeroPaddedFft works.
TEST(Aec3FftDeathTest, NullZeroPaddedFftOutput) {
  Aec3Fft fft;
  std::array<float, kFftLengthBy2> x;
  EXPECT_DEATH(fft.ZeroPaddedFft(x, Aec3Fft::Window::kRectangular, nullptr),
               "");
}

// Verifies that the check for input length in ZeroPaddedFft works.
TEST(Aec3FftDeathTest, ZeroPaddedFftWrongInputLength) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLengthBy2 - 1> x;
  EXPECT_DEATH(fft.ZeroPaddedFft(x, Aec3Fft::Window::kRectangular, &X), "");
}

// Verifies that the check for non-null output in PaddedFft works.
TEST(Aec3FftDeathTest, NullPaddedFftOutput) {
  Aec3Fft fft;
  std::array<float, kFftLengthBy2> x;
  std::array<float, kFftLengthBy2> x_old;
  EXPECT_DEATH(fft.PaddedFft(x, x_old, nullptr), "");
}

// Verifies that the check for input length in PaddedFft works.
TEST(Aec3FftDeathTest, PaddedFftWrongInputLength) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLengthBy2 - 1> x;
  std::array<float, kFftLengthBy2> x_old;
  EXPECT_DEATH(fft.PaddedFft(x, x_old, &X), "");
}

// Verifies that the check for length in the old value in PaddedFft works.
TEST(Aec3FftDeathTest, PaddedFftWrongOldValuesLength) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLengthBy2> x;
  std::array<float, kFftLengthBy2 - 1> x_old;
  EXPECT_DEATH(fft.PaddedFft(x, x_old, &X), "");
}

#endif

// Verifies that Fft works as intended.
TEST(Aec3Fft, Fft) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLength> x;
  x.fill(0.f);
  fft.Fft(&x, &X);
  EXPECT_THAT(X.re, ::testing::Each(0.f));
  EXPECT_THAT(X.im, ::testing::Each(0.f));

  x.fill(0.f);
  x[0] = 1.f;
  fft.Fft(&x, &X);
  EXPECT_THAT(X.re, ::testing::Each(1.f));
  EXPECT_THAT(X.im, ::testing::Each(0.f));

  x.fill(1.f);
  fft.Fft(&x, &X);
  EXPECT_EQ(128.f, X.re[0]);
  std::for_each(X.re.begin() + 1, X.re.end(),
                [](float a) { EXPECT_EQ(0.f, a); });
  EXPECT_THAT(X.im, ::testing::Each(0.f));
}

// Verifies that InverseFft works as intended.
TEST(Aec3Fft, Ifft) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLength> x;

  X.re.fill(0.f);
  X.im.fill(0.f);
  fft.Ifft(X, &x);
  EXPECT_THAT(x, ::testing::Each(0.f));

  X.re.fill(1.f);
  X.im.fill(0.f);
  fft.Ifft(X, &x);
  EXPECT_EQ(64.f, x[0]);
  std::for_each(x.begin() + 1, x.end(), [](float a) { EXPECT_EQ(0.f, a); });

  X.re.fill(0.f);
  X.re[0] = 128;
  X.im.fill(0.f);
  fft.Ifft(X, &x);
  EXPECT_THAT(x, ::testing::Each(64.f));
}

// Verifies that InverseFft and Fft work as intended.
TEST(Aec3Fft, FftAndIfft) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLength> x;
  std::array<float, kFftLength> x_ref;

  int v = 0;
  for (int k = 0; k < 20; ++k) {
    for (size_t j = 0; j < x.size(); ++j) {
      x[j] = v++;
      x_ref[j] = x[j] * 64.f;
    }
    fft.Fft(&x, &X);
    fft.Ifft(X, &x);
    for (size_t j = 0; j < x.size(); ++j) {
      EXPECT_NEAR(x_ref[j], x[j], 0.001f);
    }
  }
}

// Verifies that ZeroPaddedFft work as intended.
TEST(Aec3Fft, ZeroPaddedFft) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLengthBy2> x_in;
  std::array<float, kFftLength> x_ref;
  std::array<float, kFftLength> x_out;

  int v = 0;
  x_ref.fill(0.f);
  for (int k = 0; k < 20; ++k) {
    for (size_t j = 0; j < x_in.size(); ++j) {
      x_in[j] = v++;
      x_ref[j + kFftLengthBy2] = x_in[j] * 64.f;
    }
    fft.ZeroPaddedFft(x_in, Aec3Fft::Window::kRectangular, &X);
    fft.Ifft(X, &x_out);
    for (size_t j = 0; j < x_out.size(); ++j) {
      EXPECT_NEAR(x_ref[j], x_out[j], 0.1f);
    }
  }
}

// Verifies that ZeroPaddedFft work as intended.
TEST(Aec3Fft, PaddedFft) {
  Aec3Fft fft;
  FftData X;
  std::array<float, kFftLengthBy2> x_in;
  std::array<float, kFftLength> x_out;
  std::array<float, kFftLengthBy2> x_old;
  std::array<float, kFftLengthBy2> x_old_ref;
  std::array<float, kFftLength> x_ref;

  int v = 0;
  x_old.fill(0.f);
  for (int k = 0; k < 20; ++k) {
    for (size_t j = 0; j < x_in.size(); ++j) {
      x_in[j] = v++;
    }

    std::copy(x_old.begin(), x_old.end(), x_ref.begin());
    std::copy(x_in.begin(), x_in.end(), x_ref.begin() + kFftLengthBy2);
    std::copy(x_in.begin(), x_in.end(), x_old_ref.begin());
    std::for_each(x_ref.begin(), x_ref.end(), [](float& a) { a *= 64.f; });

    fft.PaddedFft(x_in, x_old, &X);
    std::copy(x_in.begin(), x_in.end(), x_old.begin());
    fft.Ifft(X, &x_out);

    for (size_t j = 0; j < x_out.size(); ++j) {
      EXPECT_NEAR(x_ref[j], x_out[j], 0.1f);
    }

    EXPECT_EQ(x_old_ref, x_old);
  }
}

}  // namespace webrtc
