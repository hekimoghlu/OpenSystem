/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "api/video/i444_buffer.h"

#include <cstdint>
#include <cstring>

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "test/frame_utils.h"
#include "test/gtest.h"

namespace webrtc {

namespace {
int GetY(rtc::scoped_refptr<I444BufferInterface> buf, int col, int row) {
  return buf->DataY()[row * buf->StrideY() + col];
}

int GetU(rtc::scoped_refptr<I444BufferInterface> buf, int col, int row) {
  return buf->DataU()[row * buf->StrideU() + col];
}

int GetV(rtc::scoped_refptr<I444BufferInterface> buf, int col, int row) {
  return buf->DataV()[row * buf->StrideV() + col];
}

void FillI444Buffer(rtc::scoped_refptr<I444Buffer> buf) {
  const uint8_t Y = 1;
  const uint8_t U = 2;
  const uint8_t V = 3;
  for (int row = 0; row < buf->height(); ++row) {
    for (int col = 0; col < buf->width(); ++col) {
      buf->MutableDataY()[row * buf->StrideY() + col] = Y;
      buf->MutableDataU()[row * buf->StrideU() + col] = U;
      buf->MutableDataV()[row * buf->StrideV() + col] = V;
    }
  }
}

}  // namespace

TEST(I444BufferTest, InitialData) {
  constexpr int stride = 3;
  constexpr int width = 3;
  constexpr int height = 3;

  rtc::scoped_refptr<I444Buffer> i444_buffer(I444Buffer::Create(width, height));
  EXPECT_EQ(width, i444_buffer->width());
  EXPECT_EQ(height, i444_buffer->height());
  EXPECT_EQ(stride, i444_buffer->StrideY());
  EXPECT_EQ(stride, i444_buffer->StrideU());
  EXPECT_EQ(stride, i444_buffer->StrideV());
  EXPECT_EQ(3, i444_buffer->ChromaWidth());
  EXPECT_EQ(3, i444_buffer->ChromaHeight());
}

TEST(I444BufferTest, ReadPixels) {
  constexpr int width = 3;
  constexpr int height = 3;

  rtc::scoped_refptr<I444Buffer> i444_buffer(I444Buffer::Create(width, height));
  // Y = 1, U = 2, V = 3.
  FillI444Buffer(i444_buffer);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      EXPECT_EQ(1, GetY(i444_buffer, col, row));
      EXPECT_EQ(2, GetU(i444_buffer, col, row));
      EXPECT_EQ(3, GetV(i444_buffer, col, row));
    }
  }
}

TEST(I444BufferTest, ToI420) {
  constexpr int width = 3;
  constexpr int height = 3;
  constexpr int size_y = width * height;
  constexpr int size_u = (width + 1) / 2 * (height + 1) / 2;
  constexpr int size_v = (width + 1) / 2 * (height + 1) / 2;
  rtc::scoped_refptr<I420Buffer> reference(I420Buffer::Create(width, height));
  memset(reference->MutableDataY(), 8, size_y);
  memset(reference->MutableDataU(), 4, size_u);
  memset(reference->MutableDataV(), 2, size_v);

  rtc::scoped_refptr<I444Buffer> i444_buffer(I444Buffer::Create(width, height));
  // Convert the reference buffer to I444.
  memset(i444_buffer->MutableDataY(), 8, size_y);
  memset(i444_buffer->MutableDataU(), 4, size_y);
  memset(i444_buffer->MutableDataV(), 2, size_y);

  // Confirm YUV values are as expected.
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      EXPECT_EQ(8, GetY(i444_buffer, col, row));
      EXPECT_EQ(4, GetU(i444_buffer, col, row));
      EXPECT_EQ(2, GetV(i444_buffer, col, row));
    }
  }

  rtc::scoped_refptr<I420BufferInterface> i420_buffer(i444_buffer->ToI420());
  EXPECT_EQ(height, i420_buffer->height());
  EXPECT_EQ(width, i420_buffer->width());
  EXPECT_TRUE(test::FrameBufsEqual(reference, i420_buffer));
}

}  // namespace webrtc
