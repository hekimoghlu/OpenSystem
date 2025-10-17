/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#include "test/hiprec_convolve_test_util.h"

using libaom_test::ACMRandom;
#if CONFIG_AV1_HIGHBITDEPTH
using libaom_test::AV1HighbdHiprecConvolve::AV1HighbdHiprecConvolveTest;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HighbdHiprecConvolveTest);
#endif
using libaom_test::AV1HiprecConvolve::AV1HiprecConvolveTest;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HiprecConvolveTest);
using std::make_tuple;
using std::tuple;

namespace {

TEST_P(AV1HiprecConvolveTest, CheckOutput) { RunCheckOutput(GET_PARAM(3)); }
TEST_P(AV1HiprecConvolveTest, DISABLED_SpeedTest) {
  RunSpeedTest(GET_PARAM(3));
}
#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(SSE2, AV1HiprecConvolveTest,
                         libaom_test::AV1HiprecConvolve::BuildParams(
                             av1_wiener_convolve_add_src_sse2));
#endif
#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, AV1HiprecConvolveTest,
                         libaom_test::AV1HiprecConvolve::BuildParams(
                             av1_wiener_convolve_add_src_avx2));
#endif
#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, AV1HiprecConvolveTest,
                         libaom_test::AV1HiprecConvolve::BuildParams(
                             av1_wiener_convolve_add_src_neon));
#endif

#if CONFIG_AV1_HIGHBITDEPTH
#if HAVE_SSSE3 || HAVE_AVX2 || HAVE_NEON
TEST_P(AV1HighbdHiprecConvolveTest, CheckOutput) {
  RunCheckOutput(GET_PARAM(4));
}
TEST_P(AV1HighbdHiprecConvolveTest, DISABLED_SpeedTest) {
  RunSpeedTest(GET_PARAM(4));
}
#if HAVE_SSSE3
INSTANTIATE_TEST_SUITE_P(SSSE3, AV1HighbdHiprecConvolveTest,
                         libaom_test::AV1HighbdHiprecConvolve::BuildParams(
                             av1_highbd_wiener_convolve_add_src_ssse3));
#endif
#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, AV1HighbdHiprecConvolveTest,
                         libaom_test::AV1HighbdHiprecConvolve::BuildParams(
                             av1_highbd_wiener_convolve_add_src_avx2));
#endif

#if HAVE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, AV1HighbdHiprecConvolveTest,
                         libaom_test::AV1HighbdHiprecConvolve::BuildParams(
                             av1_highbd_wiener_convolve_add_src_neon));
#endif
#endif
#endif  // CONFIG_AV1_HIGHBITDEPTH

}  // namespace
