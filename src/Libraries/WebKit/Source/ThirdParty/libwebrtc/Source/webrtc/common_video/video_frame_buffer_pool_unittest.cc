/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#include "common_video/include/video_frame_buffer_pool.h"

#include <stdint.h>
#include <string.h>

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "test/gtest.h"

namespace webrtc {

TEST(TestVideoFrameBufferPool, SimpleFrameReuse) {
  VideoFrameBufferPool pool;
  auto buffer = pool.CreateI420Buffer(16, 16);
  EXPECT_EQ(16, buffer->width());
  EXPECT_EQ(16, buffer->height());
  // Extract non-refcounted pointers for testing.
  const uint8_t* y_ptr = buffer->DataY();
  const uint8_t* u_ptr = buffer->DataU();
  const uint8_t* v_ptr = buffer->DataV();
  // Release buffer so that it is returned to the pool.
  buffer = nullptr;
  // Check that the memory is resued.
  buffer = pool.CreateI420Buffer(16, 16);
  EXPECT_EQ(y_ptr, buffer->DataY());
  EXPECT_EQ(u_ptr, buffer->DataU());
  EXPECT_EQ(v_ptr, buffer->DataV());
}

TEST(TestVideoFrameBufferPool, FailToReuseWrongSize) {
  // Set max frames to 1, just to make sure the first buffer is being released.
  VideoFrameBufferPool pool(/*zero_initialize=*/false, 1);
  auto buffer = pool.CreateI420Buffer(16, 16);
  EXPECT_EQ(16, buffer->width());
  EXPECT_EQ(16, buffer->height());
  // Release buffer so that it is returned to the pool.
  buffer = nullptr;
  // Check that the pool doesn't try to reuse buffers of incorrect size.
  buffer = pool.CreateI420Buffer(32, 16);
  ASSERT_TRUE(buffer);
  EXPECT_EQ(32, buffer->width());
  EXPECT_EQ(16, buffer->height());
}

TEST(TestVideoFrameBufferPool, FrameValidAfterPoolDestruction) {
  rtc::scoped_refptr<I420Buffer> buffer;
  {
    VideoFrameBufferPool pool;
    buffer = pool.CreateI420Buffer(16, 16);
  }
  EXPECT_EQ(16, buffer->width());
  EXPECT_EQ(16, buffer->height());
  // Access buffer, so that ASAN could find any issues if buffer
  // doesn't outlive the buffer pool.
  memset(buffer->MutableDataY(), 0xA5, 16 * buffer->StrideY());
}

TEST(TestVideoFrameBufferPool, MaxNumberOfBuffers) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateI420Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
  EXPECT_EQ(nullptr, pool.CreateI420Buffer(16, 16).get());
}

TEST(TestVideoFrameBufferPool, ProducesNv12) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateNV12Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
}

TEST(TestVideoFrameBufferPool, ProducesI422) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateI422Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
}

TEST(TestVideoFrameBufferPool, ProducesI444) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateI444Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
}

TEST(TestVideoFrameBufferPool, ProducesI010) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateI010Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
}

TEST(TestVideoFrameBufferPool, ProducesI210) {
  VideoFrameBufferPool pool(false, 1);
  auto buffer = pool.CreateI210Buffer(16, 16);
  EXPECT_NE(nullptr, buffer.get());
}

TEST(TestVideoFrameBufferPool, SwitchingPixelFormat) {
  VideoFrameBufferPool pool(false, 1);
  auto buffeNV12 = pool.CreateNV12Buffer(16, 16);
  EXPECT_EQ(nullptr, pool.CreateNV12Buffer(16, 16).get());

  auto bufferI420 = pool.CreateI420Buffer(16, 16);
  EXPECT_NE(nullptr, bufferI420.get());
  EXPECT_EQ(nullptr, pool.CreateI420Buffer(16, 16).get());

  auto bufferI444 = pool.CreateI444Buffer(16, 16);
  EXPECT_NE(nullptr, bufferI444.get());
  EXPECT_EQ(nullptr, pool.CreateI444Buffer(16, 16).get());

  auto bufferI422 = pool.CreateI422Buffer(16, 16);
  EXPECT_NE(nullptr, bufferI422.get());
  EXPECT_EQ(nullptr, pool.CreateI422Buffer(16, 16).get());

  auto bufferI010 = pool.CreateI010Buffer(16, 16);
  EXPECT_NE(nullptr, bufferI010.get());
  EXPECT_EQ(nullptr, pool.CreateI010Buffer(16, 16).get());

  auto bufferI210 = pool.CreateI210Buffer(16, 16);
  EXPECT_NE(nullptr, bufferI210.get());
  EXPECT_EQ(nullptr, pool.CreateI210Buffer(16, 16).get());
}

}  // namespace webrtc
