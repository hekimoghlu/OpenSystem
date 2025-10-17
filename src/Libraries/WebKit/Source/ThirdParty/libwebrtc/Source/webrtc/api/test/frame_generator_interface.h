/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#ifndef API_TEST_FRAME_GENERATOR_INTERFACE_H_
#define API_TEST_FRAME_GENERATOR_INTERFACE_H_

#include <cstddef>
#include <optional>
#include <utility>

#include "api/scoped_refptr.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"

namespace webrtc {
namespace test {

class FrameGeneratorInterface {
 public:
  struct Resolution {
    size_t width;
    size_t height;
  };
  struct VideoFrameData {
    VideoFrameData(rtc::scoped_refptr<VideoFrameBuffer> buffer,
                   std::optional<VideoFrame::UpdateRect> update_rect)
        : buffer(std::move(buffer)), update_rect(update_rect) {}

    rtc::scoped_refptr<VideoFrameBuffer> buffer;
    std::optional<VideoFrame::UpdateRect> update_rect;
  };

  enum class OutputType { kI420, kI420A, kI010, kNV12 };
  static const char* OutputTypeToString(OutputType type);

  virtual ~FrameGeneratorInterface() = default;

  // Returns VideoFrameBuffer and area where most of update was done to set them
  // on the VideoFrame object.
  virtual VideoFrameData NextFrame() = 0;
  // Skips the next frame in case it doesn't need to be encoded.
  // Default implementation is to call NextFrame and ignore the returned value.
  virtual void SkipNextFrame() { NextFrame(); }

  // Change the capture resolution.
  virtual void ChangeResolution(size_t width, size_t height) = 0;

  virtual Resolution GetResolution() const = 0;

  // Returns the frames per second this generator is supposed to provide
  // according to its data source. Not all frame generators know the frames per
  // second of the data source, in such case this method returns std::nullopt.
  virtual std::optional<int> fps() const = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_FRAME_GENERATOR_INTERFACE_H_
