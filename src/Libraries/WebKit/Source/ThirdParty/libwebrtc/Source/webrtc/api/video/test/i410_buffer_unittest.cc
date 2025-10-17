/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#include "api/video/i410_buffer.h"

#include <cstdint>
#include <cstring>

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "test/frame_utils.h"
#include "test/gtest.h"

namespace webrtc {

namespace {
constexpr uint16_t kYValue = 4;
constexpr uint16_t kUValue = 8;
constexpr uint16_t kVValue = 16;

int GetY(rtc::scoped_refptr<I410BufferInterface> buf, int col, int row) {
  return buf->DataY()[row * buf->StrideY() + col];
}

int GetU(rtc::scoped_refptr<I410BufferInterface> buf, int col, int row) {
  return buf->DataU()[row * buf->StrideU() + col];
}

int GetV(rtc::scoped_refptr<I410BufferInterface> buf, int col, int row) {
  return buf->DataV()[row * buf->StrideV() + col];
}

void FillI410Buffer(rtc::scoped_refptr<I410Buffer> buf) {
  for (int row = 0; row < buf->height(); ++row) {
    for (int col = 0; col < buf->width(); ++col) {
      buf->MutableDataY()[row * buf->StrideY() + col] = kYValue;
      buf->MutableDataU()[row * buf->StrideU() + col] = kUValue;
      buf->MutableDataV()[row * buf->StrideV() + col] = kVValue;
    }
  }
}

}  // namespace

TEST(I410BufferTest, InitialData) {
  constexpr int stride = 3;
  constexpr int width = 3;
  constexpr int height = 3;

  rtc::scoped_refptr<I410Buffer> i410_buffer(I410Buffer::Create(width, height));
  EXPECT_EQ(width, i410_buffer->width());
  EXPECT_EQ(height, i410_buffer->height());
  EXPECT_EQ(stride, i410_buffer->StrideY());
  EXPECT_EQ(stride, i410_buffer->StrideU());
  EXPECT_EQ(stride, i410_buffer->StrideV());
  EXPECT_EQ(3, i410_buffer->ChromaWidth());
  EXPECT_EQ(3, i410_buffer->ChromaHeight());
}

TEST(I410BufferTest, ReadPixels) {
  constexpr int width = 3;
  constexpr int height = 3;

  rtc::scoped_refptr<I410Buffer> i410_buffer(I410Buffer::Create(width, height));
  FillI410Buffer(i410_buffer);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      EXPECT_EQ(kYValue, GetY(i410_buffer, col, row));
      EXPECT_EQ(kUValue, GetU(i410_buffer, col, row));
      EXPECT_EQ(kVValue, GetV(i410_buffer, col, row));
    }
  }
}

TEST(I410BufferTest, ToI420) {
  // libyuv I410ToI420 only handles correctly even sizes and skips last row/col
  // if odd.
  constexpr int width = 4;
  constexpr int height = 4;
  constexpr int size_y = width * height;
  constexpr int size_u = (width + 1) / 2 * (height + 1) / 2;
  constexpr int size_v = (width + 1) / 2 * (height + 1) / 2;
  rtc::scoped_refptr<I420Buffer> reference(I420Buffer::Create(width, height));
  // I410 is 10-bit while I420 is 8 bit, so last 2 bits would be discarded.
  memset(reference->MutableDataY(), kYValue >> 2, size_y);
  memset(reference->MutableDataU(), kUValue >> 2, size_u);
  memset(reference->MutableDataV(), kVValue >> 2, size_v);

  rtc::scoped_refptr<I410Buffer> i410_buffer(I410Buffer::Create(width, height));
  FillI410Buffer(i410_buffer);

  // Confirm YUV values are as expected.
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      EXPECT_EQ(kYValue, GetY(i410_buffer, col, row));
      EXPECT_EQ(kUValue, GetU(i410_buffer, col, row));
      EXPECT_EQ(kVValue, GetV(i410_buffer, col, row));
    }
  }

  rtc::scoped_refptr<I420BufferInterface> i420_buffer(i410_buffer->ToI420());

  // Confirm YUV values are as expected.
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      EXPECT_EQ(1, i420_buffer->DataY()[row * i420_buffer->StrideY() + col]);
    }
  }

  EXPECT_EQ(height, i420_buffer->height());
  EXPECT_EQ(width, i420_buffer->width());
  EXPECT_TRUE(test::FrameBufsEqual(reference, i420_buffer));
}

}  // namespace webrtc
