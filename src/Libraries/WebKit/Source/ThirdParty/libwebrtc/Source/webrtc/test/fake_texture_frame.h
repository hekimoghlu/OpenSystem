/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef TEST_FAKE_TEXTURE_FRAME_H_
#define TEST_FAKE_TEXTURE_FRAME_H_

#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
#include "common_video/include/video_frame_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

class FakeNativeBuffer : public VideoFrameBuffer {
 public:
  static VideoFrame CreateFrame(int width,
                                int height,
                                uint32_t timestamp,
                                int64_t render_time_ms,
                                VideoRotation rotation);

  FakeNativeBuffer(int width, int height) : width_(width), height_(height) {}

  Type type() const override;
  int width() const override;
  int height() const override;

 private:
  rtc::scoped_refptr<I420BufferInterface> ToI420() override;

  const int width_;
  const int height_;
};

}  // namespace test
}  // namespace webrtc
#endif  //  TEST_FAKE_TEXTURE_FRAME_H_
