/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "modules/audio_processing/aec3/vector_math.h"

#include <math.h>

#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"
#include "test/gtest.h"

namespace webrtc {

#if defined(WEBRTC_HAS_NEON)

TEST(VectorMath, Sqrt) {
  std::array<float, kFftLengthBy2Plus1> x;
  std::array<float, kFftLengthBy2Plus1> z;
  std::array<float, kFftLengthBy2Plus1> z_neon;

  for (size_t k = 0; k < x.size(); ++k) {
    x[k] = (2.f / 3.f) * k;
  }

  std::copy(x.begin(), x.end(), z.begin());
  aec3::VectorMath(Aec3Optimization::kNone).Sqrt(z);
  std::copy(x.begin(), x.end(), z_neon.begin());
  aec3::VectorMath(Aec3Optimization::kNeon).Sqrt(z_neon);
  for (size_t k = 0; k < z.size(); ++k) {
    EXPECT_NEAR(z[k], z_neon[k], 0.0001f);
    EXPECT_NEAR(sqrtf(x[k]), z_neon[k], 0.0001f);
  }
}

TEST(VectorMath, Multiply) {
  std::array<float, kFftLengthBy2Plus1> x;
  std::array<float, kFftLengthBy2Plus1> y;
  std::array<float, kFftLengthBy2Plus1> z;
  std::array<float, kFftLengthBy2Plus1> z_neon;

  for (size_t k = 0; k < x.size(); ++k) {
    x[k] = k;
    y[k] = (2.f / 3.f) * k;
  }

  aec3::VectorMath(Aec3Optimization::kNone).Multiply(x, y, z);
  aec3::VectorMath(Aec3Optimization::kNeon).Multiply(x, y, z_neon);
  for (size_t k = 0; k < z.size(); ++k) {
    EXPECT_FLOAT_EQ(z[k], z_neon[k]);
    EXPECT_FLOAT_EQ(x[k] * y[k], z_neon[k]);
  }
}

TEST(VectorMath, Accumulate) {
  std::array<float, kFftLengthBy2Plus1> x;
  std::array<float, kFftLengthBy2Plus1> z;
  std::array<float, kFftLengthBy2Plus1> z_neon;

  for (size_t k = 0; k < x.size(); ++k) {
    x[k] = k;
    z[k] = z_neon[k] = 2.f * k;
  }

  aec3::VectorMath(Aec3Optimization::kNone).Accumulate(x, z);
  aec3::VectorMath(Aec3Optimization::kNeon).Accumulate(x, z_neon);
  for (size_t k = 0; k < z.size(); ++k) {
    EXPECT_FLOAT_EQ(z[k], z_neon[k]);
    EXPECT_FLOAT_EQ(x[k] + 2.f * x[k], z_neon[k]);
  }
}
#endif

#if defined(WEBRTC_ARCH_X86_FAMILY)

TEST(VectorMath, Sse2Sqrt) {
  if (GetCPUInfo(kSSE2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_sse2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = (2.f / 3.f) * k;
    }

    std::copy(x.begin(), x.end(), z.begin());
    aec3::VectorMath(Aec3Optimization::kNone).Sqrt(z);
    std::copy(x.begin(), x.end(), z_sse2.begin());
    aec3::VectorMath(Aec3Optimization::kSse2).Sqrt(z_sse2);
    EXPECT_EQ(z, z_sse2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_sse2[k]);
      EXPECT_FLOAT_EQ(sqrtf(x[k]), z_sse2[k]);
    }
  }
}

TEST(VectorMath, Avx2Sqrt) {
  if (GetCPUInfo(kAVX2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_avx2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = (2.f / 3.f) * k;
    }

    std::copy(x.begin(), x.end(), z.begin());
    aec3::VectorMath(Aec3Optimization::kNone).Sqrt(z);
    std::copy(x.begin(), x.end(), z_avx2.begin());
    aec3::VectorMath(Aec3Optimization::kAvx2).Sqrt(z_avx2);
    EXPECT_EQ(z, z_avx2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_avx2[k]);
      EXPECT_FLOAT_EQ(sqrtf(x[k]), z_avx2[k]);
    }
  }
}

TEST(VectorMath, Sse2Multiply) {
  if (GetCPUInfo(kSSE2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> y;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_sse2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = k;
      y[k] = (2.f / 3.f) * k;
    }

    aec3::VectorMath(Aec3Optimization::kNone).Multiply(x, y, z);
    aec3::VectorMath(Aec3Optimization::kSse2).Multiply(x, y, z_sse2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_sse2[k]);
      EXPECT_FLOAT_EQ(x[k] * y[k], z_sse2[k]);
    }
  }
}

TEST(VectorMath, Avx2Multiply) {
  if (GetCPUInfo(kAVX2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> y;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_avx2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = k;
      y[k] = (2.f / 3.f) * k;
    }

    aec3::VectorMath(Aec3Optimization::kNone).Multiply(x, y, z);
    aec3::VectorMath(Aec3Optimization::kAvx2).Multiply(x, y, z_avx2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_avx2[k]);
      EXPECT_FLOAT_EQ(x[k] * y[k], z_avx2[k]);
    }
  }
}

TEST(VectorMath, Sse2Accumulate) {
  if (GetCPUInfo(kSSE2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_sse2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = k;
      z[k] = z_sse2[k] = 2.f * k;
    }

    aec3::VectorMath(Aec3Optimization::kNone).Accumulate(x, z);
    aec3::VectorMath(Aec3Optimization::kSse2).Accumulate(x, z_sse2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_sse2[k]);
      EXPECT_FLOAT_EQ(x[k] + 2.f * x[k], z_sse2[k]);
    }
  }
}

TEST(VectorMath, Avx2Accumulate) {
  if (GetCPUInfo(kAVX2) != 0) {
    std::array<float, kFftLengthBy2Plus1> x;
    std::array<float, kFftLengthBy2Plus1> z;
    std::array<float, kFftLengthBy2Plus1> z_avx2;

    for (size_t k = 0; k < x.size(); ++k) {
      x[k] = k;
      z[k] = z_avx2[k] = 2.f * k;
    }

    aec3::VectorMath(Aec3Optimization::kNone).Accumulate(x, z);
    aec3::VectorMath(Aec3Optimization::kAvx2).Accumulate(x, z_avx2);
    for (size_t k = 0; k < z.size(); ++k) {
      EXPECT_FLOAT_EQ(z[k], z_avx2[k]);
      EXPECT_FLOAT_EQ(x[k] + 2.f * x[k], z_avx2[k]);
    }
  }
}
#endif

}  // namespace webrtc
