/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#ifndef MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_V4L2_H_
#define MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_V4L2_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "modules/video_capture/video_capture_defines.h"
#include "modules/video_capture/video_capture_impl.h"
#include "rtc_base/platform_thread.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace videocapturemodule {
class VideoCaptureModuleV4L2 : public VideoCaptureImpl {
 public:
  VideoCaptureModuleV4L2();
  ~VideoCaptureModuleV4L2() override;
  int32_t Init(const char* deviceUniqueId);
  int32_t StartCapture(const VideoCaptureCapability& capability) override;
  int32_t StopCapture() override;
  bool CaptureStarted() override;
  int32_t CaptureSettings(VideoCaptureCapability& settings) override;

 private:
  enum { kNoOfV4L2Bufffers = 4 };

  static void CaptureThread(void*);
  bool CaptureProcess();
  bool AllocateVideoBuffers() RTC_EXCLUSIVE_LOCKS_REQUIRED(capture_lock_);
  bool DeAllocateVideoBuffers() RTC_EXCLUSIVE_LOCKS_REQUIRED(capture_lock_);

  rtc::PlatformThread _captureThread RTC_GUARDED_BY(api_checker_);
  Mutex capture_lock_ RTC_ACQUIRED_BEFORE(api_lock_);
  bool quit_ RTC_GUARDED_BY(capture_lock_);
  int32_t _deviceId RTC_GUARDED_BY(api_checker_);
  int32_t _deviceFd RTC_GUARDED_BY(capture_checker_);

  int32_t _buffersAllocatedByDevice RTC_GUARDED_BY(capture_lock_);
  VideoCaptureCapability configured_capability_
      RTC_GUARDED_BY(capture_checker_);
  bool _streaming RTC_GUARDED_BY(capture_checker_);
  bool _captureStarted RTC_GUARDED_BY(api_checker_);
  struct Buffer {
    void* start;
    size_t length;
  };
  Buffer* _pool RTC_GUARDED_BY(capture_lock_);
};
}  // namespace videocapturemodule
}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_V4L2_H_
