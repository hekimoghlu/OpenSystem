/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#ifndef COMMON_VIDEO_FRAMERATE_CONTROLLER_H_
#define COMMON_VIDEO_FRAMERATE_CONTROLLER_H_

#include <stdint.h>

#include <optional>

namespace webrtc {

// Determines which frames that should be dropped based on input framerate and
// requested framerate.
class FramerateController {
 public:
  FramerateController();
  explicit FramerateController(double max_framerate);
  ~FramerateController();

  // Sets max framerate (default is maxdouble).
  void SetMaxFramerate(double max_framerate);
  double GetMaxFramerate() const;

  // Returns true if the frame should be dropped, false otherwise.
  bool ShouldDropFrame(int64_t in_timestamp_ns);

  void Reset();

  void KeepFrame(int64_t in_timestamp_ns);

 private:
  double max_framerate_;
  std::optional<int64_t> next_frame_timestamp_ns_;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_FRAMERATE_CONTROLLER_H_
