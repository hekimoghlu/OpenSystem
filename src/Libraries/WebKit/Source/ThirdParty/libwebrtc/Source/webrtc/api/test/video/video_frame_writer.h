/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#ifndef API_TEST_VIDEO_VIDEO_FRAME_WRITER_H_
#define API_TEST_VIDEO_VIDEO_FRAME_WRITER_H_

#include "api/video/video_frame.h"

namespace webrtc {
namespace test {

class VideoFrameWriter {
 public:
  virtual ~VideoFrameWriter() = default;

  // Writes `VideoFrame` and returns true if operation was successful, false
  // otherwise.
  //
  // Calling `WriteFrame` after `Close` is not allowed.
  virtual bool WriteFrame(const VideoFrame& frame) = 0;

  // Closes writer and cleans up all resources. No invocations to `WriteFrame`
  // are allowed after `Close` was invoked.
  virtual void Close() = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_VIDEO_VIDEO_FRAME_WRITER_H_
