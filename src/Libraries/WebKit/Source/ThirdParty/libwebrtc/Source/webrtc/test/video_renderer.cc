/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "test/video_renderer.h"

// TODO(pbos): Android renderer

namespace webrtc {
namespace test {

class NullRenderer : public VideoRenderer {
  void OnFrame(const VideoFrame& video_frame) override {}
};

VideoRenderer* VideoRenderer::Create(const char* window_title,
                                     size_t width,
                                     size_t height) {
  VideoRenderer* renderer = CreatePlatformRenderer(window_title, width, height);
  if (renderer != nullptr)
    return renderer;

  return new NullRenderer();
}
}  // namespace test
}  // namespace webrtc
