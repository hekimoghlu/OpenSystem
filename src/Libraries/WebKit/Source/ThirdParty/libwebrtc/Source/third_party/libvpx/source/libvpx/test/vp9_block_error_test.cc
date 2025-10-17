/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vp9_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_entropy.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

using libvpx_test::ACMRandom;

namespace {
const int kNumIterations = 1000;

typedef int64_t (*HBDBlockErrorFunc)(const tran_low_t *coeff,
                                     const tran_low_t *dqcoeff,
                                     intptr_t block_size, int64_t *ssz,
                                     int bps);

typedef std::tuple<HBDBlockErrorFunc, HBDBlockErrorFunc, vpx_bit_depth_t>
    BlockErrorParam;

typedef int64_t (*BlockErrorFunc)(const tran_low_t *coeff,
                                  const tran_low_t *dqcoeff,
                                  intptr_t block_size, int64_t *ssz);

template <BlockErrorFunc fn>
int64_t BlockError8BitWrapper(const tran_low_t *coeff,
                              const tran_low_t *dqcoeff, intptr_t block_size,
                              int64_t *ssz, int bps) {
  EXPECT_EQ(bps, 8);
  return fn(coeff, dqcoeff, block_size, ssz);
}

class BlockErrorTest : public ::testing::TestWithParam<BlockErrorParam> {
 public:
  ~BlockErrorTest() override = default;
  void SetUp() override {
    error_block_op_ = GET_PARAM(0);
    ref_error_block_op_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  vpx_bit_depth_t bit_depth_;
  HBDBlockErrorFunc error_block_op_;
  HBDBlockErrorFunc ref_error_block_op_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BlockErrorTest);

TEST_P(BlockErrorTest, OperationCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, tran_low_t, coeff[4096]);
  DECLARE_ALIGNED(16, tran_low_t, dqcoeff[4096]);
  int err_count_total = 0;
  int first_failure = -1;
  intptr_t block_size;
  int64_t ssz;
  int64_t ret;
  int64_t ref_ssz;
  int64_t ref_ret;
  const int msb = bit_depth_ + 8 - 1;
  for (int i = 0; i < kNumIterations; ++i) {
    int err_count = 0;
    block_size = 16 << (i % 9);  // All block sizes from 4x4, 8x4 ..64x64
    for (int j = 0; j < block_size; j++) {
      // coeff and dqcoeff will always have at least the same sign, and this
      // can be used for optimization, so generate test input precisely.
      if (rnd(2)) {
        // Positive number
        coeff[j] = rnd(1 << msb);
        dqcoeff[j] = rnd(1 << msb);
      } else {
        // Negative number
        coeff[j] = -rnd(1 << msb);
        dqcoeff[j] = -rnd(1 << msb);
      }
    }
    ref_ret =
        ref_error_block_op_(coeff, dqcoeff, block_size, &ref_ssz, bit_depth_);
    ASM_REGISTER_STATE_CHECK(
        ret = error_block_op_(coeff, dqcoeff, block_size, &ssz, bit_depth_));
    err_count += (ref_ret != ret) | (ref_ssz != ssz);
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Error Block Test, C output doesn't match optimized output. "
      << "First failed at test case " << first_failure;
}

TEST_P(BlockErrorTest, ExtremeValues) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, tran_low_t, coeff[4096]);
  DECLARE_ALIGNED(16, tran_low_t, dqcoeff[4096]);
  int err_count_total = 0;
  int first_failure = -1;
  intptr_t block_size;
  int64_t ssz;
  int64_t ret;
  int64_t ref_ssz;
  int64_t ref_ret;
  const int msb = bit_depth_ + 8 - 1;
  int max_val = ((1 << msb) - 1);
  for (int i = 0; i < kNumIterations; ++i) {
    int err_count = 0;
    int k = (i / 9) % 9;

    // Change the maximum coeff value, to test different bit boundaries
    if (k == 8 && (i % 9) == 0) {
      max_val >>= 1;
    }
    block_size = 16 << (i % 9);  // All block sizes from 4x4, 8x4 ..64x64
    for (int j = 0; j < block_size; j++) {
      if (k < 4) {
        // Test at positive maximum values
        coeff[j] = k % 2 ? max_val : 0;
        dqcoeff[j] = (k >> 1) % 2 ? max_val : 0;
      } else if (k < 8) {
        // Test at negative maximum values
        coeff[j] = k % 2 ? -max_val : 0;
        dqcoeff[j] = (k >> 1) % 2 ? -max_val : 0;
      } else {
        if (rnd(2)) {
          // Positive number
          coeff[j] = rnd(1 << 14);
          dqcoeff[j] = rnd(1 << 14);
        } else {
          // Negative number
          coeff[j] = -rnd(1 << 14);
          dqcoeff[j] = -rnd(1 << 14);
        }
      }
    }
    ref_ret =
        ref_error_block_op_(coeff, dqcoeff, block_size, &ref_ssz, bit_depth_);
    ASM_REGISTER_STATE_CHECK(
        ret = error_block_op_(coeff, dqcoeff, block_size, &ssz, bit_depth_));
    err_count += (ref_ret != ret) | (ref_ssz != ssz);
    if (err_count && !err_count_total) {
      first_failure = i;
    }
    err_count_total += err_count;
  }
  EXPECT_EQ(0, err_count_total)
      << "Error: Error Block Test, C output doesn't match optimized output. "
      << "First failed at test case " << first_failure;
}

using std::make_tuple;

#if HAVE_SSE2
const BlockErrorParam sse2_block_error_tests[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vp9_highbd_block_error_sse2, &vp9_highbd_block_error_c,
             VPX_BITS_10),
  make_tuple(&vp9_highbd_block_error_sse2, &vp9_highbd_block_error_c,
             VPX_BITS_12),
  make_tuple(&vp9_highbd_block_error_sse2, &vp9_highbd_block_error_c,
             VPX_BITS_8),
#endif  // CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&BlockError8BitWrapper<vp9_block_error_sse2>,
             &BlockError8BitWrapper<vp9_block_error_c>, VPX_BITS_8)
};

INSTANTIATE_TEST_SUITE_P(SSE2, BlockErrorTest,
                         ::testing::ValuesIn(sse2_block_error_tests));
#endif  // HAVE_SSE2

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, BlockErrorTest,
    ::testing::Values(make_tuple(&BlockError8BitWrapper<vp9_block_error_avx2>,
                                 &BlockError8BitWrapper<vp9_block_error_c>,
                                 VPX_BITS_8)));
#endif  // HAVE_AVX2

#if HAVE_NEON
const BlockErrorParam neon_block_error_tests[] = {
#if CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&vp9_highbd_block_error_neon, &vp9_highbd_block_error_c,
             VPX_BITS_10),
  make_tuple(&vp9_highbd_block_error_neon, &vp9_highbd_block_error_c,
             VPX_BITS_12),
  make_tuple(&vp9_highbd_block_error_neon, &vp9_highbd_block_error_c,
             VPX_BITS_8),
#endif  // CONFIG_VP9_HIGHBITDEPTH
  make_tuple(&BlockError8BitWrapper<vp9_block_error_neon>,
             &BlockError8BitWrapper<vp9_block_error_c>, VPX_BITS_8)
};

INSTANTIATE_TEST_SUITE_P(NEON, BlockErrorTest,
                         ::testing::ValuesIn(neon_block_error_tests));
#endif  // HAVE_NEON

#if HAVE_SVE
const BlockErrorParam sve_block_error_tests[] = { make_tuple(
    &BlockError8BitWrapper<vp9_block_error_sve>,
    &BlockError8BitWrapper<vp9_block_error_c>, VPX_BITS_8) };

INSTANTIATE_TEST_SUITE_P(SVE, BlockErrorTest,
                         ::testing::ValuesIn(sve_block_error_tests));
#endif  // HAVE_SVE
}  // namespace
