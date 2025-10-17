/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef TEST_VIDEO_RENDERER_H_
#define TEST_VIDEO_RENDERER_H_

#include <stddef.h>

#include "api/video/video_sink_interface.h"

namespace webrtc {
class VideoFrame;

namespace test {
class VideoRenderer : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  // Creates a platform-specific renderer if possible, or a null implementation
  // if failing.
  static VideoRenderer* Create(const char* window_title,
                               size_t width,
                               size_t height);
  // Returns a renderer rendering to a platform specific window if possible,
  // NULL if none can be created.
  // Creates a platform-specific renderer if possible, returns NULL if a
  // platform renderer could not be created. This occurs, for instance, when
  // running without an X environment on Linux.
  static VideoRenderer* CreatePlatformRenderer(const char* window_title,
                                               size_t width,
                                               size_t height);
  virtual ~VideoRenderer() {}

 protected:
  VideoRenderer() {}
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_VIDEO_RENDERER_H_
