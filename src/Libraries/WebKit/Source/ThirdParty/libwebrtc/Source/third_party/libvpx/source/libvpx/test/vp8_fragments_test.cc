/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include "test/codec_factory.h"
#include "test/video_source.h"

namespace {

class VP8FragmentsTest : public ::libvpx_test::EncoderTest,
                         public ::testing::Test {
 protected:
  VP8FragmentsTest() : EncoderTest(&::libvpx_test::kVP8) {}
  ~VP8FragmentsTest() override = default;

  void SetUp() override {
    const unsigned long init_flags =  // NOLINT(runtime/int)
        VPX_CODEC_USE_OUTPUT_PARTITION;
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    set_init_flags(init_flags);
  }
};

TEST_F(VP8FragmentsTest, TestFragmentsEncodeDecode) {
  ::libvpx_test::RandomVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

}  // namespace
