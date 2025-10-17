/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#ifndef TEST_FRAME_GENERATOR_CAPTURER_H_
#define TEST_FRAME_GENERATOR_CAPTURER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/test/frame_generator_interface.h"
#include "api/video/color_space.h"
#include "api/video/video_frame.h"
#include "api/video/video_rotation.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"
#include "test/test_video_capturer.h"

namespace webrtc {
namespace test {

class FrameGeneratorCapturer : public TestVideoCapturer {
 public:
  class SinkWantsObserver {
   public:
    // OnSinkWantsChanged is called when FrameGeneratorCapturer::AddOrUpdateSink
    // is called.
    virtual void OnSinkWantsChanged(rtc::VideoSinkInterface<VideoFrame>* sink,
                                    const rtc::VideoSinkWants& wants) = 0;

   protected:
    virtual ~SinkWantsObserver() {}
  };

  FrameGeneratorCapturer(
      Clock* clock,
      std::unique_ptr<FrameGeneratorInterface> frame_generator,
      int target_fps,
      TaskQueueFactory& task_queue_factory);
  virtual ~FrameGeneratorCapturer();

  void Start() override;
  void Stop() override;
  void ChangeResolution(size_t width, size_t height);
  void ChangeFramerate(int target_framerate);

  int GetFrameWidth() const override;
  int GetFrameHeight() const override;

  struct Resolution {
    int width;
    int height;
  };
  std::optional<Resolution> GetResolution() const;

  void OnOutputFormatRequest(int width,
                             int height,
                             const std::optional<int>& max_fps);

  void SetSinkWantsObserver(SinkWantsObserver* observer);

  void AddOrUpdateSink(rtc::VideoSinkInterface<VideoFrame>* sink,
                       const rtc::VideoSinkWants& wants) override;
  void RemoveSink(rtc::VideoSinkInterface<VideoFrame>* sink) override;

  void ForceFrame();
  void SetFakeRotation(VideoRotation rotation);
  void SetFakeColorSpace(std::optional<ColorSpace> color_space);

  bool Init();

 private:
  void InsertFrame();
  static bool Run(void* obj);
  int GetCurrentConfiguredFramerate();

  Clock* const clock_;
  RepeatingTaskHandle frame_task_;
  bool sending_;
  SinkWantsObserver* sink_wants_observer_ RTC_GUARDED_BY(&lock_);

  Mutex lock_;
  std::unique_ptr<FrameGeneratorInterface> frame_generator_;

  int source_fps_ RTC_GUARDED_BY(&lock_);
  int target_capture_fps_ RTC_GUARDED_BY(&lock_);
  VideoRotation fake_rotation_ = kVideoRotation_0;
  std::optional<ColorSpace> fake_color_space_ RTC_GUARDED_BY(&lock_);

  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> task_queue_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_FRAME_GENERATOR_CAPTURER_H_
