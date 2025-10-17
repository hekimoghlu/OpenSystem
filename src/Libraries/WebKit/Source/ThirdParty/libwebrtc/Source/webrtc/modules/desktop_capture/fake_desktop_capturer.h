/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_FAKE_DESKTOP_CAPTURER_H_
#define MODULES_DESKTOP_CAPTURE_FAKE_DESKTOP_CAPTURER_H_

#include <memory>

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame_generator.h"
#include "modules/desktop_capture/shared_memory.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A fake implementation of DesktopCapturer or its derived interfaces to
// generate DesktopFrame for testing purpose.
//
// Consumers can provide a FrameGenerator instance to generate instances of
// DesktopFrame to return for each Capture() function call.
// If no FrameGenerator provided, FakeDesktopCapturer will always return a
// nullptr DesktopFrame.
//
// Double buffering is guaranteed by the FrameGenerator. FrameGenerator
// implements in desktop_frame_generator.h guarantee double buffering, they
// creates a new instance of DesktopFrame each time.
class RTC_EXPORT FakeDesktopCapturer : public DesktopCapturer {
 public:
  FakeDesktopCapturer();
  ~FakeDesktopCapturer() override;

  // Decides the result which will be returned in next Capture() callback.
  void set_result(DesktopCapturer::Result result);

  // Uses the `generator` provided as DesktopFrameGenerator, FakeDesktopCapturer
  // does not take the ownership of `generator`.
  void set_frame_generator(DesktopFrameGenerator* generator);

  // Count of DesktopFrame(s) have been returned by this instance. This field
  // would never be negative.
  int num_frames_captured() const;

  // Count of CaptureFrame() calls have been made. This field would never be
  // negative.
  int num_capture_attempts() const;

  // DesktopCapturer interface
  void Start(DesktopCapturer::Callback* callback) override;
  void CaptureFrame() override;
  void SetSharedMemoryFactory(
      std::unique_ptr<SharedMemoryFactory> shared_memory_factory) override;
  bool GetSourceList(DesktopCapturer::SourceList* sources) override;
  bool SelectSource(DesktopCapturer::SourceId id) override;

 private:
  static constexpr DesktopCapturer::SourceId kWindowId = 1378277495;
  static constexpr DesktopCapturer::SourceId kScreenId = 1378277496;

  DesktopCapturer::Callback* callback_ = nullptr;
  std::unique_ptr<SharedMemoryFactory> shared_memory_factory_;
  DesktopCapturer::Result result_ = Result::SUCCESS;
  DesktopFrameGenerator* generator_ = nullptr;
  int num_frames_captured_ = 0;
  int num_capture_attempts_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_FAKE_DESKTOP_CAPTURER_H_
