/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#ifndef PC_TEST_FAKE_VIDEO_TRACK_SOURCE_H_
#define PC_TEST_FAKE_VIDEO_TRACK_SOURCE_H_

#include "api/media_stream_interface.h"
#include "media/base/video_broadcaster.h"
#include "pc/video_track_source.h"

namespace webrtc {

// A minimal implementation of VideoTrackSource. Includes a VideoBroadcaster for
// injection of frames.
class FakeVideoTrackSource : public VideoTrackSource {
 public:
  static rtc::scoped_refptr<FakeVideoTrackSource> Create(bool is_screencast) {
    return rtc::make_ref_counted<FakeVideoTrackSource>(is_screencast);
  }

  static rtc::scoped_refptr<FakeVideoTrackSource> Create() {
    return Create(false);
  }

  bool is_screencast() const override { return is_screencast_; }

  void InjectFrame(const VideoFrame& frame) {
    video_broadcaster_.OnFrame(frame);
  }

 protected:
  explicit FakeVideoTrackSource(bool is_screencast)
      : VideoTrackSource(false /* remote */), is_screencast_(is_screencast) {}
  ~FakeVideoTrackSource() override = default;

  rtc::VideoSourceInterface<VideoFrame>* source() override {
    return &video_broadcaster_;
  }

 private:
  const bool is_screencast_;
  rtc::VideoBroadcaster video_broadcaster_;
};

}  // namespace webrtc

#endif  // PC_TEST_FAKE_VIDEO_TRACK_SOURCE_H_
