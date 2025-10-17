/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#ifndef TEST_FRAME_UTILS_H_
#define TEST_FRAME_UTILS_H_

#include <stdint.h>

#include "api/scoped_refptr.h"
#include "api/video/nv12_buffer.h"

namespace webrtc {
class I420Buffer;
class VideoFrame;
class VideoFrameBuffer;
namespace test {

bool EqualPlane(const uint8_t* data1,
                const uint8_t* data2,
                int stride1,
                int stride2,
                int width,
                int height);

static inline bool EqualPlane(const uint8_t* data1,
                              const uint8_t* data2,
                              int stride,
                              int width,
                              int height) {
  return EqualPlane(data1, data2, stride, stride, width, height);
}

bool FramesEqual(const webrtc::VideoFrame& f1, const webrtc::VideoFrame& f2);

bool FrameBufsEqual(const rtc::scoped_refptr<webrtc::VideoFrameBuffer>& f1,
                    const rtc::scoped_refptr<webrtc::VideoFrameBuffer>& f2);

rtc::scoped_refptr<I420Buffer> ReadI420Buffer(int width, int height, FILE*);

rtc::scoped_refptr<NV12Buffer> ReadNV12Buffer(int width, int height, FILE*);

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FRAME_UTILS_H_
