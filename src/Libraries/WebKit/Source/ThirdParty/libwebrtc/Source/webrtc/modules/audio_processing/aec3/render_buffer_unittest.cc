/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include "modules/audio_processing/aec3/render_buffer.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "test/gtest.h"

namespace webrtc {

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

// Verifies the check for non-null fft buffer.
TEST(RenderBufferDeathTest, NullExternalFftBuffer) {
  BlockBuffer block_buffer(10, 3, 1);
  SpectrumBuffer spectrum_buffer(10, 1);
  EXPECT_DEATH(RenderBuffer(&block_buffer, &spectrum_buffer, nullptr), "");
}

// Verifies the check for non-null spectrum buffer.
TEST(RenderBufferDeathTest, NullExternalSpectrumBuffer) {
  FftBuffer fft_buffer(10, 1);
  BlockBuffer block_buffer(10, 3, 1);
  EXPECT_DEATH(RenderBuffer(&block_buffer, nullptr, &fft_buffer), "");
}

// Verifies the check for non-null block buffer.
TEST(RenderBufferDeathTest, NullExternalBlockBuffer) {
  FftBuffer fft_buffer(10, 1);
  SpectrumBuffer spectrum_buffer(10, 1);
  EXPECT_DEATH(RenderBuffer(nullptr, &spectrum_buffer, &fft_buffer), "");
}

#endif

}  // namespace webrtc
