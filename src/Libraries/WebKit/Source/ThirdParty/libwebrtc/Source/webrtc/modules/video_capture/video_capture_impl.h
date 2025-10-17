/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_IMPL_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_IMPL_H_

/*
 * video_capture_impl.h
 */

#include <stddef.h>
#include <stdint.h>

#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "api/video/video_frame.h"
#include "api/video/video_rotation.h"
#include "api/video/video_sink_interface.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_config.h"
#include "modules/video_capture/video_capture_defines.h"
#include "rtc_base/race_checker.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class VideoCaptureOptions;

namespace videocapturemodule {
// Class definitions
class RTC_EXPORT VideoCaptureImpl : public VideoCaptureModule {
 public:
  /*
   *   Create a video capture module object
   *
   *   id              - unique identifier of this video capture module object
   *   deviceUniqueIdUTF8 -  name of the device. Available names can be found by
   * using GetDeviceName
   */
  static rtc::scoped_refptr<VideoCaptureModule> Create(
      const char* deviceUniqueIdUTF8);
  static rtc::scoped_refptr<VideoCaptureModule> Create(
      VideoCaptureOptions* options,
      const char* deviceUniqueIdUTF8);

  static DeviceInfo* CreateDeviceInfo();
  static DeviceInfo* CreateDeviceInfo(VideoCaptureOptions* options);

  // Helpers for converting between (integral) degrees and
  // VideoRotation values.  Return 0 on success.
  static int32_t RotationFromDegrees(int degrees, VideoRotation* rotation);
  static int32_t RotationInDegrees(VideoRotation rotation, int* degrees);

  // Call backs
  void RegisterCaptureDataCallback(
      rtc::VideoSinkInterface<VideoFrame>* dataCallback) override;
  virtual void RegisterCaptureDataCallback(
      RawVideoSinkInterface* dataCallback) override;
  void DeRegisterCaptureDataCallback() override;

  int32_t SetCaptureRotation(VideoRotation rotation) override;
  bool SetApplyRotation(bool enable) override;
  bool GetApplyRotation() override;

  const char* CurrentDeviceName() const override;

  // `capture_time` must be specified in NTP time format in milliseconds.
  int32_t IncomingFrame(uint8_t* videoFrame,
                        size_t videoFrameLength,
                        const VideoCaptureCapability& frameInfo,
                        int64_t captureTime = 0);

  // Platform dependent
  int32_t StartCapture(const VideoCaptureCapability& capability) override;
  int32_t StopCapture() override;
  bool CaptureStarted() override;
  int32_t CaptureSettings(VideoCaptureCapability& /*settings*/) override;

 protected:
  VideoCaptureImpl();
  ~VideoCaptureImpl() override;

  // Calls to the public API must happen on a single thread.
  SequenceChecker api_checker_;
  // RaceChecker for members that can be accessed on the API thread while
  // capture is not happening, and on a callback thread otherwise.
  rtc::RaceChecker capture_checker_;
  // current Device unique name;
  char* _deviceUniqueId RTC_GUARDED_BY(api_checker_);
  Mutex api_lock_;
  // Should be set by platform dependent code in StartCapture.
  VideoCaptureCapability _requestedCapability RTC_GUARDED_BY(api_checker_);

 private:
  void UpdateFrameCount();
  uint32_t CalculateFrameRate(int64_t now_ns);
  int32_t DeliverCapturedFrame(VideoFrame& captureFrame)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(api_lock_);
  void DeliverRawFrame(uint8_t* videoFrame,
                       size_t videoFrameLength,
                       const VideoCaptureCapability& frameInfo,
                       int64_t captureTime)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(api_lock_);

  // last time the module process function was called.
  int64_t _lastProcessTimeNanos RTC_GUARDED_BY(capture_checker_);
  // last time the frame rate callback function was called.
  int64_t _lastFrameRateCallbackTimeNanos RTC_GUARDED_BY(capture_checker_);

  rtc::VideoSinkInterface<VideoFrame>* _dataCallBack RTC_GUARDED_BY(api_lock_);
  RawVideoSinkInterface* _rawDataCallBack RTC_GUARDED_BY(api_lock_);

  int64_t _lastProcessFrameTimeNanos RTC_GUARDED_BY(capture_checker_);
  // timestamp for local captured frames
  int64_t _incomingFrameTimesNanos[kFrameRateCountHistorySize] RTC_GUARDED_BY(
      capture_checker_);
  // Set if the frame should be rotated by the capture module.
  VideoRotation _rotateFrame RTC_GUARDED_BY(api_lock_);

  // Indicate whether rotation should be applied before delivered externally.
  bool apply_rotation_ RTC_GUARDED_BY(api_lock_);
};
}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_VIDEO_CAPTURE_IMPL_H_
