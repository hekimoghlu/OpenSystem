/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#include <tuple>

#include "gtest/gtest.h"
#include "test/warp_filter_test_util.h"
using libaom_test::ACMRandom;
#if CONFIG_AV1_HIGHBITDEPTH
using libaom_test::AV1HighbdWarpFilter::AV1HighbdWarpFilterTest;
#endif
using libaom_test::AV1WarpFilter::AV1WarpFilterTest;
using std::make_tuple;
using std::tuple;

namespace {

TEST_P(AV1WarpFilterTest, CheckOutput) {
  RunCheckOutput(std::get<3>(GET_PARAM(0)));
}
TEST_P(AV1WarpFilterTest, DISABLED_Speed) {
  RunSpeedTest(std::get<3>(GET_PARAM(0)));
}

INSTANTIATE_TEST_SUITE_P(
    C, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_c));

#if CONFIG_AV1_HIGHBITDEPTH && (HAVE_SSE4_1 || HAVE_NEON)
TEST_P(AV1HighbdWarpFilterTest, CheckOutput) {
  RunCheckOutput(std::get<4>(GET_PARAM(0)));
}
TEST_P(AV1HighbdWarpFilterTest, DISABLED_Speed) {
  RunSpeedTest(std::get<4>(GET_PARAM(0)));
}
#endif  // CONFIG_AV1_HIGHBITDEPTH && (HAVE_SSE4_1 || HAVE_NEON)

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_sse4_1));

#if CONFIG_AV1_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(SSE4_1, AV1HighbdWarpFilterTest,
                         libaom_test::AV1HighbdWarpFilter::BuildParams(
                             av1_highbd_warp_affine_sse4_1));
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // HAVE_SSE4_1

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_avx2));

#if CONFIG_AV1_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1HighbdWarpFilterTest,
    libaom_test::AV1HighbdWarpFilter::BuildParams(av1_highbd_warp_affine_avx2));
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // HAVE_AVX2

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_neon));

#if CONFIG_AV1_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1HighbdWarpFilterTest,
    libaom_test::AV1HighbdWarpFilter::BuildParams(av1_highbd_warp_affine_neon));
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // HAVE_NEON

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(
    NEON_I8MM, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_neon_i8mm));
#endif  // HAVE_NEON_I8MM

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, AV1WarpFilterTest,
    libaom_test::AV1WarpFilter::BuildParams(av1_warp_affine_sve));

#if CONFIG_AV1_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    SVE, AV1HighbdWarpFilterTest,
    libaom_test::AV1HighbdWarpFilter::BuildParams(av1_highbd_warp_affine_sve));
#endif  // CONFIG_AV1_HIGHBITDEPTH
#endif  // HAVE_SVE

}  // namespace
