/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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
#ifndef MEDIA_BASE_FAKE_VIDEO_RENDERER_H_
#define MEDIA_BASE_FAKE_VIDEO_RENDERER_H_

#include <stdint.h>

#include "api/scoped_refptr.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"
#include "api/video/video_rotation.h"
#include "api/video/video_sink_interface.h"
#include "rtc_base/synchronization/mutex.h"

namespace cricket {

// Faked video renderer that has a callback for actions on rendering.
class FakeVideoRenderer : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
 public:
  FakeVideoRenderer();

  void OnFrame(const webrtc::VideoFrame& frame) override;

  int width() const {
    webrtc::MutexLock lock(&mutex_);
    return width_;
  }
  int height() const {
    webrtc::MutexLock lock(&mutex_);
    return height_;
  }

  webrtc::VideoRotation rotation() const {
    webrtc::MutexLock lock(&mutex_);
    return rotation_;
  }

  int64_t timestamp_us() const {
    webrtc::MutexLock lock(&mutex_);
    return timestamp_us_;
  }

  int num_rendered_frames() const {
    webrtc::MutexLock lock(&mutex_);
    return num_rendered_frames_;
  }

  bool black_frame() const {
    webrtc::MutexLock lock(&mutex_);
    return black_frame_;
  }

 private:
  int width_ = 0;
  int height_ = 0;
  webrtc::VideoRotation rotation_ = webrtc::kVideoRotation_0;
  int64_t timestamp_us_ = 0;
  int num_rendered_frames_ = 0;
  bool black_frame_ = false;
  mutable webrtc::Mutex mutex_;
};

}  // namespace cricket

#endif  // MEDIA_BASE_FAKE_VIDEO_RENDERER_H_
