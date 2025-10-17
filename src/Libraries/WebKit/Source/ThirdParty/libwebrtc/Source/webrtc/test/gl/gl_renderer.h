/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef TEST_GL_GL_RENDERER_H_
#define TEST_GL_GL_RENDERER_H_

#ifdef WEBRTC_MAC
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <stddef.h>
#include <stdint.h>

#include "api/video/video_frame.h"
#include "test/video_renderer.h"

namespace webrtc {
namespace test {

class GlRenderer : public VideoRenderer {
 public:
  void OnFrame(const webrtc::VideoFrame& frame) override;

 protected:
  GlRenderer();

  void Init();
  void Destroy();

  void ResizeViewport(size_t width, size_t height);

 private:
  bool is_init_;
  uint8_t* buffer_;
  GLuint texture_;
  size_t width_, height_, buffer_size_;

  void ResizeVideo(size_t width, size_t height);
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_GL_GL_RENDERER_H_
