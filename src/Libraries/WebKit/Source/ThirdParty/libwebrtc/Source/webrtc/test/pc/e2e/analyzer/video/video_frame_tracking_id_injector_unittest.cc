/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "test/pc/e2e/analyzer/video/video_frame_tracking_id_injector.h"

#include "api/video/encoded_image.h"
#include "rtc_base/buffer.h"
#include "test/gtest.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

EncodedImage CreateEncodedImageOfSizeN(size_t n) {
  EncodedImage image;
  rtc::scoped_refptr<EncodedImageBuffer> buffer = EncodedImageBuffer::Create(n);
  for (size_t i = 0; i < n; ++i) {
    buffer->data()[i] = static_cast<uint8_t>(i);
  }
  image.SetEncodedData(buffer);
  return image;
}

TEST(VideoFrameTrackingIdInjectorTest, InjectExtractDiscardFalse) {
  VideoFrameTrackingIdInjector injector;
  EncodedImage source = CreateEncodedImageOfSizeN(10);
  EncodedImageExtractionResult out =
      injector.ExtractData(injector.InjectData(512, false, source));

  ASSERT_TRUE(out.id.has_value());
  EXPECT_EQ(*out.id, 512);
  EXPECT_FALSE(out.discard);
  EXPECT_EQ(out.image.size(), 10ul);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(source.data()[i], out.image.data()[i]);
  }
}

#if GTEST_HAS_DEATH_TEST
TEST(VideoFrameTrackingIdInjectorTest, InjectExtractDiscardTrue) {
  VideoFrameTrackingIdInjector injector;
  EncodedImage source = CreateEncodedImageOfSizeN(10);

  EXPECT_DEATH(injector.InjectData(512, true, source), "");
}
#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace webrtc_pc_e2e
}  // namespace webrtc
