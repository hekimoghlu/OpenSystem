/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#ifndef TEST_LINUX_GLX_RENDERER_H_
#define TEST_LINUX_GLX_RENDERER_H_

#include <GL/glx.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <stddef.h>

#include "api/video/video_frame.h"
#include "test/gl/gl_renderer.h"

namespace webrtc {
namespace test {

class GlxRenderer : public GlRenderer {
 public:
  static GlxRenderer* Create(const char* window_title,
                             size_t width,
                             size_t height);
  virtual ~GlxRenderer();

  void OnFrame(const webrtc::VideoFrame& frame) override;

 private:
  GlxRenderer(size_t width, size_t height);

  bool Init(const char* window_title);
  void Resize(size_t width, size_t height);
  void Destroy();

  size_t width_, height_;

  Display* display_;
  Window window_;
  GLXContext context_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_LINUX_GLX_RENDERER_H_
