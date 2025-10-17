/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#ifndef TEST_TEST_VIDEO_CAPTURER_H_
#define TEST_TEST_VIDEO_CAPTURER_H_

#include <stddef.h>

#include <memory>

#include "api/video/video_frame.h"
#include "api/video/video_source_interface.h"
#include "media/base/video_adapter.h"
#include "media/base/video_broadcaster.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace test {

class TestVideoCapturer : public rtc::VideoSourceInterface<VideoFrame> {
 public:
  class FramePreprocessor {
   public:
    virtual ~FramePreprocessor() = default;

    virtual VideoFrame Preprocess(const VideoFrame& frame) = 0;
  };

  ~TestVideoCapturer() override;

  void AddOrUpdateSink(rtc::VideoSinkInterface<VideoFrame>* sink,
                       const rtc::VideoSinkWants& wants) override;
  void RemoveSink(rtc::VideoSinkInterface<VideoFrame>* sink) override;
  void SetFramePreprocessor(std::unique_ptr<FramePreprocessor> preprocessor) {
    MutexLock lock(&lock_);
    preprocessor_ = std::move(preprocessor);
  }
  void SetEnableAdaptation(bool enable_adaptation) {
    MutexLock lock(&lock_);
    enable_adaptation_ = enable_adaptation;
  }
  void OnOutputFormatRequest(int width,
                             int height,
                             const std::optional<int>& max_fps);

  // Starts or resumes video capturing. Can be called multiple times during
  // lifetime of this object.
  virtual void Start() = 0;
  // Stops or pauses video capturing. Can be called multiple times during
  // lifetime of this object.
  virtual void Stop() = 0;

  virtual int GetFrameWidth() const = 0;
  virtual int GetFrameHeight() const = 0;

 protected:
  void OnFrame(const VideoFrame& frame);
  rtc::VideoSinkWants GetSinkWants();

 private:
  void UpdateVideoAdapter();
  VideoFrame MaybePreprocess(const VideoFrame& frame);

  Mutex lock_;
  std::unique_ptr<FramePreprocessor> preprocessor_ RTC_GUARDED_BY(lock_);
  bool enable_adaptation_ RTC_GUARDED_BY(lock_) = true;
  rtc::VideoBroadcaster broadcaster_;
  cricket::VideoAdapter video_adapter_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_TEST_VIDEO_CAPTURER_H_
