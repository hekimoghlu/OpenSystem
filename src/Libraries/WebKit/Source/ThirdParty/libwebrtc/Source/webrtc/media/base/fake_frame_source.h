/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#ifndef MEDIA_BASE_FAKE_FRAME_SOURCE_H_
#define MEDIA_BASE_FAKE_FRAME_SOURCE_H_

#include "api/video/video_frame.h"
#include "rtc_base/time_utils.h"

namespace cricket {

class FakeFrameSource {
 public:
  FakeFrameSource(int width,
                  int height,
                  int interval_us,
                  int64_t timestamp_offset_us);
  FakeFrameSource(int width, int height, int interval_us);

  webrtc::VideoRotation GetRotation() const;
  void SetRotation(webrtc::VideoRotation rotation);

  webrtc::VideoFrame GetFrame();
  webrtc::VideoFrame GetFrameRotationApplied();

  // Override configuration.
  webrtc::VideoFrame GetFrame(int width,
                              int height,
                              webrtc::VideoRotation rotation,
                              int interval_us);

 private:
  const int width_;
  const int height_;
  const int interval_us_;

  webrtc::VideoRotation rotation_ = webrtc::kVideoRotation_0;
  int64_t next_timestamp_us_;
};

}  // namespace cricket

#endif  // MEDIA_BASE_FAKE_FRAME_SOURCE_H_
