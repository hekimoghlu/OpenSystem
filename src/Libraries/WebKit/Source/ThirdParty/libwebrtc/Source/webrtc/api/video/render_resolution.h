/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#ifndef API_VIDEO_RENDER_RESOLUTION_H_
#define API_VIDEO_RENDER_RESOLUTION_H_

namespace webrtc {

// TODO(bugs.webrtc.org/12114) : remove in favor of Resolution.
class RenderResolution {
 public:
  constexpr RenderResolution() = default;
  constexpr RenderResolution(int width, int height)
      : width_(width), height_(height) {}
  RenderResolution(const RenderResolution&) = default;
  RenderResolution& operator=(const RenderResolution&) = default;

  friend bool operator==(const RenderResolution& lhs,
                         const RenderResolution& rhs) {
    return lhs.width_ == rhs.width_ && lhs.height_ == rhs.height_;
  }
  friend bool operator!=(const RenderResolution& lhs,
                         const RenderResolution& rhs) {
    return !(lhs == rhs);
  }

  constexpr bool Valid() const { return width_ > 0 && height_ > 0; }

  constexpr int Width() const { return width_; }
  constexpr int Height() const { return height_; }

 private:
  int width_ = 0;
  int height_ = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_RENDER_RESOLUTION_H_
