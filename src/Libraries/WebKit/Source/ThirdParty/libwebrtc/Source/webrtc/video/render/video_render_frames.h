/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#ifndef VIDEO_RENDER_VIDEO_RENDER_FRAMES_H_
#define VIDEO_RENDER_VIDEO_RENDER_FRAMES_H_

#include <stddef.h>
#include <stdint.h>

#include <list>
#include <optional>

#include "api/video/video_frame.h"

namespace webrtc {

// Class definitions
class VideoRenderFrames {
 public:
  explicit VideoRenderFrames(uint32_t render_delay_ms);
  VideoRenderFrames(const VideoRenderFrames&) = delete;
  ~VideoRenderFrames();

  // Add a frame to the render queue
  int32_t AddFrame(VideoFrame&& new_frame);

  // Get a frame for rendering, or false if it's not time to render.
  std::optional<VideoFrame> FrameToRender();

  // Returns the number of ms to next frame to render
  uint32_t TimeToNextFrameRelease();

  bool HasPendingFrames() const;

 private:
  // Sorted list with framed to be rendered, oldest first.
  std::list<VideoFrame> incoming_frames_;

  // Estimated delay from a frame is released until it's rendered.
  const uint32_t render_delay_ms_;

  int64_t last_render_time_ms_ = 0;
  size_t frames_dropped_ = 0;
};

}  // namespace webrtc

#endif  // VIDEO_RENDER_VIDEO_RENDER_FRAMES_H_
