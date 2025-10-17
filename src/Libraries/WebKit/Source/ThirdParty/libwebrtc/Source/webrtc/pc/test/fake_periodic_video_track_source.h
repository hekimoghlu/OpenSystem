/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#ifndef PC_TEST_FAKE_PERIODIC_VIDEO_TRACK_SOURCE_H_
#define PC_TEST_FAKE_PERIODIC_VIDEO_TRACK_SOURCE_H_

#include "pc/test/fake_periodic_video_source.h"
#include "pc/video_track_source.h"

namespace webrtc {

// A VideoTrackSource generating frames with configured size and frame interval.
class FakePeriodicVideoTrackSource : public VideoTrackSource {
 public:
  explicit FakePeriodicVideoTrackSource(bool remote)
      : FakePeriodicVideoTrackSource(FakePeriodicVideoSource::Config(),
                                     remote) {}

  FakePeriodicVideoTrackSource(FakePeriodicVideoSource::Config config,
                               bool remote)
      : VideoTrackSource(remote), source_(config) {}

  ~FakePeriodicVideoTrackSource() = default;

  FakePeriodicVideoSource& fake_periodic_source() { return source_; }
  const FakePeriodicVideoSource& fake_periodic_source() const {
    return source_;
  }

 protected:
  rtc::VideoSourceInterface<VideoFrame>* source() override { return &source_; }

 private:
  FakePeriodicVideoSource source_;
};

}  // namespace webrtc

#endif  // PC_TEST_FAKE_PERIODIC_VIDEO_TRACK_SOURCE_H_
