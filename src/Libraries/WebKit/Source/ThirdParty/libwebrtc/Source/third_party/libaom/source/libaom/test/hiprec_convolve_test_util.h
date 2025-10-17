/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef AOM_TEST_HIPREC_CONVOLVE_TEST_UTIL_H_
#define AOM_TEST_HIPREC_CONVOLVE_TEST_UTIL_H_

#include <tuple>

#include "config/av1_rtcd.h"

#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "test/util.h"
#include "test/register_state_check.h"

#include "aom_ports/aom_timer.h"
#include "av1/common/convolve.h"
#include "av1/common/mv.h"

namespace libaom_test {

namespace AV1HiprecConvolve {

typedef void (*hiprec_convolve_func)(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     const int16_t *filter_x, int x_step_q4,
                                     const int16_t *filter_y, int y_step_q4,
                                     int w, int h,
                                     const WienerConvolveParams *conv_params);

typedef std::tuple<int, int, int, hiprec_convolve_func> HiprecConvolveParam;

::testing::internal::ParamGenerator<HiprecConvolveParam> BuildParams(
    hiprec_convolve_func filter);

class AV1HiprecConvolveTest
    : public ::testing::TestWithParam<HiprecConvolveParam> {
 public:
  ~AV1HiprecConvolveTest() override;
  void SetUp() override;

 protected:
  void RunCheckOutput(hiprec_convolve_func test_impl);
  void RunSpeedTest(hiprec_convolve_func test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1HiprecConvolve

#if CONFIG_AV1_HIGHBITDEPTH
namespace AV1HighbdHiprecConvolve {
typedef void (*highbd_hiprec_convolve_func)(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4,
    const int16_t *filter_y, int y_step_q4, int w, int h,
    const WienerConvolveParams *conv_params, int bps);

typedef std::tuple<int, int, int, int, highbd_hiprec_convolve_func>
    HighbdHiprecConvolveParam;

::testing::internal::ParamGenerator<HighbdHiprecConvolveParam> BuildParams(
    highbd_hiprec_convolve_func filter);

class AV1HighbdHiprecConvolveTest
    : public ::testing::TestWithParam<HighbdHiprecConvolveParam> {
 public:
  ~AV1HighbdHiprecConvolveTest() override;
  void SetUp() override;

 protected:
  void RunCheckOutput(highbd_hiprec_convolve_func test_impl);
  void RunSpeedTest(highbd_hiprec_convolve_func test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1HighbdHiprecConvolve
#endif  // CONFIG_AV1_HIGHBITDEPTH
}  // namespace libaom_test

#endif  // AOM_TEST_HIPREC_CONVOLVE_TEST_UTIL_H_
