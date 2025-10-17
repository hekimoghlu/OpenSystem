/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#include "test/pc/e2e/analyzer/video/simulcast_dummy_buffer_helper.h"

#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_base/random.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

uint8_t RandByte(Random& random) {
  return random.Rand(255);
}

VideoFrame CreateRandom2x2VideoFrame(uint16_t id, Random& random) {
  rtc::scoped_refptr<I420Buffer> buffer = I420Buffer::Create(2, 2);

  uint8_t data[6] = {RandByte(random), RandByte(random), RandByte(random),
                     RandByte(random), RandByte(random), RandByte(random)};

  memcpy(buffer->MutableDataY(), data, 2);
  memcpy(buffer->MutableDataY() + buffer->StrideY(), data + 2, 2);
  memcpy(buffer->MutableDataU(), data + 4, 1);
  memcpy(buffer->MutableDataV(), data + 5, 1);

  return VideoFrame::Builder()
      .set_id(id)
      .set_video_frame_buffer(buffer)
      .set_timestamp_us(1)
      .build();
}

TEST(CreateDummyFrameBufferTest, CreatedBufferIsDummy) {
  VideoFrame dummy_frame = VideoFrame::Builder()
                               .set_video_frame_buffer(CreateDummyFrameBuffer())
                               .build();

  EXPECT_TRUE(IsDummyFrame(dummy_frame));
}

TEST(IsDummyFrameTest, NotEveryFrameIsDummy) {
  Random random(/*seed=*/100);
  VideoFrame frame = CreateRandom2x2VideoFrame(1, random);
  EXPECT_FALSE(IsDummyFrame(frame));
}

}  // namespace
}  // namespace webrtc_pc_e2e
}  // namespace webrtc
