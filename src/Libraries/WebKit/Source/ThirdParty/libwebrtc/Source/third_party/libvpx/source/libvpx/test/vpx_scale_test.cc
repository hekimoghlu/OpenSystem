/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/vpx_scale_test.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_scale/yv12config.h"

namespace libvpx_test {
namespace {

#if VPX_ARCH_ARM || (VPX_ARCH_MIPS && !HAVE_MIPS64) || VPX_ARCH_X86
// Avoid OOM failures on 32-bit platforms.
const int kNumSizesToTest = 7;
#else
const int kNumSizesToTest = 8;
#endif
const int kSizesToTest[] = { 1, 15, 33, 145, 512, 1025, 3840, 16383 };

typedef void (*ExtendFrameBorderFunc)(YV12_BUFFER_CONFIG *ybf);
typedef void (*CopyFrameFunc)(const YV12_BUFFER_CONFIG *src_ybf,
                              YV12_BUFFER_CONFIG *dst_ybf);

class ExtendBorderTest
    : public VpxScaleBase,
      public ::testing::TestWithParam<ExtendFrameBorderFunc> {
 public:
  ~ExtendBorderTest() override = default;

 protected:
  void SetUp() override { extend_fn_ = GetParam(); }

  void ExtendBorder() { ASM_REGISTER_STATE_CHECK(extend_fn_(&img_)); }

  void RunTest() {
    for (int h = 0; h < kNumSizesToTest; ++h) {
      for (int w = 0; w < kNumSizesToTest; ++w) {
        ASSERT_NO_FATAL_FAILURE(ResetImages(kSizesToTest[w], kSizesToTest[h]));
        ReferenceCopyFrame();
        ExtendBorder();
        CompareImages(img_);
        DeallocImages();
      }
    }
  }

  ExtendFrameBorderFunc extend_fn_;
};

TEST_P(ExtendBorderTest, ExtendBorder) { ASSERT_NO_FATAL_FAILURE(RunTest()); }

INSTANTIATE_TEST_SUITE_P(C, ExtendBorderTest,
                         ::testing::Values(vp8_yv12_extend_frame_borders_c));

class CopyFrameTest : public VpxScaleBase,
                      public ::testing::TestWithParam<CopyFrameFunc> {
 public:
  ~CopyFrameTest() override = default;

 protected:
  void SetUp() override { copy_frame_fn_ = GetParam(); }

  void CopyFrame() {
    ASM_REGISTER_STATE_CHECK(copy_frame_fn_(&img_, &dst_img_));
  }

  void RunTest() {
    for (int h = 0; h < kNumSizesToTest; ++h) {
      for (int w = 0; w < kNumSizesToTest; ++w) {
        ASSERT_NO_FATAL_FAILURE(ResetImages(kSizesToTest[w], kSizesToTest[h]));
        ReferenceCopyFrame();
        CopyFrame();
        CompareImages(dst_img_);
        DeallocImages();
      }
    }
  }

  CopyFrameFunc copy_frame_fn_;
};

TEST_P(CopyFrameTest, CopyFrame) { ASSERT_NO_FATAL_FAILURE(RunTest()); }

INSTANTIATE_TEST_SUITE_P(C, CopyFrameTest,
                         ::testing::Values(vp8_yv12_copy_frame_c));

}  // namespace
}  // namespace libvpx_test
