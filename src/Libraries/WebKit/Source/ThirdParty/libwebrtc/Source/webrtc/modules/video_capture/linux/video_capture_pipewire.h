/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#ifndef MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_PIPEWIRE_H_
#define MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_PIPEWIRE_H_

#include "modules/video_capture/linux/pipewire_session.h"
#include "modules/video_capture/video_capture_defines.h"
#include "modules/video_capture/video_capture_impl.h"

namespace webrtc {
namespace videocapturemodule {
class VideoCaptureModulePipeWire : public VideoCaptureImpl {
 public:
  explicit VideoCaptureModulePipeWire(VideoCaptureOptions* options);
  ~VideoCaptureModulePipeWire() override;
  int32_t Init(const char* deviceUniqueId);
  int32_t StartCapture(const VideoCaptureCapability& capability) override;
  int32_t StopCapture() override;
  bool CaptureStarted() override;
  int32_t CaptureSettings(VideoCaptureCapability& settings) override;

  static VideoType PipeWireRawFormatToVideoType(uint32_t format);
  static uint32_t VideoTypeToPipeWireRawFormat(VideoType type);

 private:
  static void OnStreamParamChanged(void* data,
                                   uint32_t id,
                                   const struct spa_pod* format);
  static void OnStreamStateChanged(void* data,
                                   pw_stream_state old_state,
                                   pw_stream_state state,
                                   const char* error_message);

  static void OnStreamProcess(void* data);

  void OnFormatChanged(const struct spa_pod* format);
  void ProcessBuffers();

  const rtc::scoped_refptr<PipeWireSession> session_
      RTC_GUARDED_BY(api_checker_);
  bool initialized_ RTC_GUARDED_BY(api_checker_);
  bool started_ RTC_GUARDED_BY(api_lock_);
  int node_id_ RTC_GUARDED_BY(capture_checker_);
  VideoCaptureCapability configured_capability_
      RTC_GUARDED_BY(capture_checker_);

  struct pw_stream* stream_ RTC_GUARDED_BY(capture_checker_) = nullptr;
  struct spa_hook stream_listener_ RTC_GUARDED_BY(capture_checker_);
};
}  // namespace videocapturemodule
}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_LINUX_VIDEO_CAPTURE_PIPEWIRE_H_
