/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>

#include "aom_dsp/aom_dsp_common.h"
#include "gtest/gtest.h"

#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/transform_test_base.h"
#include "test/util.h"
#include "av1/common/entropy.h"
#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

using libaom_test::ACMRandom;

namespace {

template <typename OutputType>
using FdctFunc = void (*)(const int16_t *in, OutputType *out, int stride);

template <typename OutputType>
using FhtFunc = void (*)(const int16_t *in, OutputType *out, int stride,
                         TxfmParam *txfm_param);

template <typename OutputType>
using Fdct4x4Param =
    std::tuple<FdctFunc<OutputType>, FhtFunc<OutputType>, aom_bit_depth_t, int>;

#if HAVE_NEON || HAVE_SSE2
void fdct4x4_ref(const int16_t *in, tran_low_t *out, int stride,
                 TxfmParam * /*txfm_param*/) {
  aom_fdct4x4_c(in, out, stride);
}

void fdct4x4_lp_ref(const int16_t *in, int16_t *out, int stride,
                    TxfmParam * /*txfm_param*/) {
  aom_fdct4x4_lp_c(in, out, stride);
}
#endif

template <typename OutputType>
class Trans4x4FDCT : public libaom_test::TransformTestBase<OutputType>,
                     public ::testing::TestWithParam<Fdct4x4Param<OutputType>> {
 public:
  ~Trans4x4FDCT() override = default;

  using TxfmBaseOutType = libaom_test::TransformTestBase<OutputType>;
  void SetUp() override {
    fwd_txfm_ = std::get<0>(this->GetParam());
    TxfmBaseOutType::pitch_ = 4;
    TxfmBaseOutType::height_ = 4;
    TxfmBaseOutType::fwd_txfm_ref = std::get<1>(this->GetParam());
    TxfmBaseOutType::bit_depth_ = std::get<2>(this->GetParam());
    TxfmBaseOutType::mask_ = (1 << TxfmBaseOutType::bit_depth_) - 1;
    TxfmBaseOutType::num_coeffs_ = std::get<3>(this->GetParam());
  }

 protected:
  void RunFwdTxfm(const int16_t *in, OutputType *out, int stride) override {
    fwd_txfm_(in, out, stride);
  }

  void RunInvTxfm(const OutputType *out, uint8_t *dst, int stride) override {
    (void)out;
    (void)dst;
    (void)stride;
  }

  FdctFunc<OutputType> fwd_txfm_;
};

using Trans4x4FDCTTranLow = Trans4x4FDCT<tran_low_t>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Trans4x4FDCTTranLow);
TEST_P(Trans4x4FDCTTranLow, CoeffCheck) { RunCoeffCheck(); }
TEST_P(Trans4x4FDCTTranLow, MemCheck) { RunMemCheck(); }

using Trans4x4FDCTInt16 = Trans4x4FDCT<int16_t>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Trans4x4FDCTInt16);
TEST_P(Trans4x4FDCTInt16, CoeffCheck) { RunCoeffCheck(); }
TEST_P(Trans4x4FDCTInt16, MemCheck) { RunMemCheck(); }

using std::make_tuple;

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, Trans4x4FDCTTranLow,
                         ::testing::Values(make_tuple(&aom_fdct4x4_neon,
                                                      &fdct4x4_ref, AOM_BITS_8,
                                                      16)));

INSTANTIATE_TEST_SUITE_P(NEON, Trans4x4FDCTInt16,
                         ::testing::Values(make_tuple(&aom_fdct4x4_lp_neon,
                                                      &fdct4x4_lp_ref,
                                                      AOM_BITS_8, 16)));
#endif

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, Trans4x4FDCTTranLow,
                         ::testing::Values(make_tuple(&aom_fdct4x4_sse2,
                                                      &fdct4x4_ref, AOM_BITS_8,
                                                      16)));

INSTANTIATE_TEST_SUITE_P(SSE2, Trans4x4FDCTInt16,
                         ::testing::Values(make_tuple(&aom_fdct4x4_lp_sse2,
                                                      &fdct4x4_lp_ref,
                                                      AOM_BITS_8, 16)));
#endif
}  // namespace
