/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#ifndef TEST_FRAME_FORWARDER_H_
#define TEST_FRAME_FORWARDER_H_

#include "api/video/video_frame.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace test {

// FrameForwarder can be used as an implementation
// of rtc::VideoSourceInterface<VideoFrame> where the caller controls when
// a frame should be forwarded to its sink.
// Currently this implementation only support one sink.
class FrameForwarder : public rtc::VideoSourceInterface<VideoFrame> {
 public:
  FrameForwarder();
  ~FrameForwarder() override;
  // Forwards `video_frame` to the registered `sink_`.
  virtual void IncomingCapturedFrame(const VideoFrame& video_frame)
      RTC_LOCKS_EXCLUDED(mutex_);
  rtc::VideoSinkWants sink_wants() const RTC_LOCKS_EXCLUDED(mutex_);
  bool has_sinks() const RTC_LOCKS_EXCLUDED(mutex_);

 protected:
  rtc::VideoSinkWants sink_wants_locked() const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void AddOrUpdateSink(rtc::VideoSinkInterface<VideoFrame>* sink,
                       const rtc::VideoSinkWants& wants)
      RTC_LOCKS_EXCLUDED(mutex_) override;
  void AddOrUpdateSinkLocked(rtc::VideoSinkInterface<VideoFrame>* sink,
                             const rtc::VideoSinkWants& wants)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void RemoveSink(rtc::VideoSinkInterface<VideoFrame>* sink)
      RTC_LOCKS_EXCLUDED(mutex_) override;

  mutable Mutex mutex_;
  rtc::VideoSinkInterface<VideoFrame>* sink_ RTC_GUARDED_BY(mutex_);
  rtc::VideoSinkWants sink_wants_ RTC_GUARDED_BY(mutex_);
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FRAME_FORWARDER_H_
