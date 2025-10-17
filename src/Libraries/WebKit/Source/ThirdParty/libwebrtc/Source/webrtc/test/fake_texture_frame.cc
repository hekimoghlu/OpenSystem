/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "test/fake_texture_frame.h"

#include "api/video/i420_buffer.h"

namespace webrtc {
namespace test {

VideoFrame FakeNativeBuffer::CreateFrame(int width,
                                         int height,
                                         uint32_t timestamp,
                                         int64_t render_time_ms,
                                         VideoRotation rotation) {
  return VideoFrame::Builder()
      .set_video_frame_buffer(
          rtc::make_ref_counted<FakeNativeBuffer>(width, height))
      .set_rtp_timestamp(timestamp)
      .set_timestamp_ms(render_time_ms)
      .set_rotation(rotation)
      .build();
}

VideoFrameBuffer::Type FakeNativeBuffer::type() const {
  return Type::kNative;
}

int FakeNativeBuffer::width() const {
  return width_;
}

int FakeNativeBuffer::height() const {
  return height_;
}

rtc::scoped_refptr<I420BufferInterface> FakeNativeBuffer::ToI420() {
  rtc::scoped_refptr<I420Buffer> buffer = I420Buffer::Create(width_, height_);
  I420Buffer::SetBlack(buffer.get());
  return buffer;
}

}  // namespace test
}  // namespace webrtc
