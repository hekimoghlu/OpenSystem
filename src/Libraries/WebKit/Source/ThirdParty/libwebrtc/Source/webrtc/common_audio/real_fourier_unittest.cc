/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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

#include <stdlib.h>

#include "common_audio/real_fourier_ooura.h"
#include "test/gtest.h"

namespace webrtc {

using std::complex;

TEST(RealFourierStaticsTest, AllocatorAlignment) {
  {
    RealFourier::fft_real_scoper real;
    real = RealFourier::AllocRealBuffer(3);
    ASSERT_TRUE(real.get() != nullptr);
    uintptr_t ptr_value = reinterpret_cast<uintptr_t>(real.get());
    EXPECT_EQ(0u, ptr_value % RealFourier::kFftBufferAlignment);
  }
  {
    RealFourier::fft_cplx_scoper cplx;
    cplx = RealFourier::AllocCplxBuffer(3);
    ASSERT_TRUE(cplx.get() != nullptr);
    uintptr_t ptr_value = reinterpret_cast<uintptr_t>(cplx.get());
    EXPECT_EQ(0u, ptr_value % RealFourier::kFftBufferAlignment);
  }
}

TEST(RealFourierStaticsTest, OrderComputation) {
  EXPECT_EQ(4, RealFourier::FftOrder(13));
  EXPECT_EQ(5, RealFourier::FftOrder(32));
  EXPECT_EQ(1, RealFourier::FftOrder(2));
  EXPECT_EQ(0, RealFourier::FftOrder(1));
}

TEST(RealFourierStaticsTest, ComplexLengthComputation) {
  EXPECT_EQ(2U, RealFourier::ComplexLength(1));
  EXPECT_EQ(3U, RealFourier::ComplexLength(2));
  EXPECT_EQ(5U, RealFourier::ComplexLength(3));
  EXPECT_EQ(9U, RealFourier::ComplexLength(4));
  EXPECT_EQ(17U, RealFourier::ComplexLength(5));
  EXPECT_EQ(65U, RealFourier::ComplexLength(7));
}

template <typename T>
class RealFourierTest : public ::testing::Test {
 protected:
  RealFourierTest()
      : rf_(2),
        real_buffer_(RealFourier::AllocRealBuffer(4)),
        cplx_buffer_(RealFourier::AllocCplxBuffer(3)) {}

  ~RealFourierTest() {}

  T rf_;
  const RealFourier::fft_real_scoper real_buffer_;
  const RealFourier::fft_cplx_scoper cplx_buffer_;
};

using FftTypes = ::testing::Types<RealFourierOoura>;
TYPED_TEST_SUITE(RealFourierTest, FftTypes);

TYPED_TEST(RealFourierTest, SimpleForwardTransform) {
  this->real_buffer_[0] = 1.0f;
  this->real_buffer_[1] = 2.0f;
  this->real_buffer_[2] = 3.0f;
  this->real_buffer_[3] = 4.0f;

  this->rf_.Forward(this->real_buffer_.get(), this->cplx_buffer_.get());

  EXPECT_NEAR(this->cplx_buffer_[0].real(), 10.0f, 1e-8f);
  EXPECT_NEAR(this->cplx_buffer_[0].imag(), 0.0f, 1e-8f);
  EXPECT_NEAR(this->cplx_buffer_[1].real(), -2.0f, 1e-8f);
  EXPECT_NEAR(this->cplx_buffer_[1].imag(), 2.0f, 1e-8f);
  EXPECT_NEAR(this->cplx_buffer_[2].real(), -2.0f, 1e-8f);
  EXPECT_NEAR(this->cplx_buffer_[2].imag(), 0.0f, 1e-8f);
}

TYPED_TEST(RealFourierTest, SimpleBackwardTransform) {
  this->cplx_buffer_[0] = complex<float>(10.0f, 0.0f);
  this->cplx_buffer_[1] = complex<float>(-2.0f, 2.0f);
  this->cplx_buffer_[2] = complex<float>(-2.0f, 0.0f);

  this->rf_.Inverse(this->cplx_buffer_.get(), this->real_buffer_.get());

  EXPECT_NEAR(this->real_buffer_[0], 1.0f, 1e-8f);
  EXPECT_NEAR(this->real_buffer_[1], 2.0f, 1e-8f);
  EXPECT_NEAR(this->real_buffer_[2], 3.0f, 1e-8f);
  EXPECT_NEAR(this->real_buffer_[3], 4.0f, 1e-8f);
}

}  // namespace webrtc
