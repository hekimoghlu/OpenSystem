/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_DEVICE_INFO_IMPL_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_DEVICE_INFO_IMPL_H_

#include <stdint.h>

#include <vector>

#include "api/video/video_rotation.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_defines.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace videocapturemodule {
class DeviceInfoImpl : public VideoCaptureModule::DeviceInfo {
 public:
  DeviceInfoImpl();
  ~DeviceInfoImpl(void) override;
  int32_t NumberOfCapabilities(const char* deviceUniqueIdUTF8) override;
  int32_t GetCapability(const char* deviceUniqueIdUTF8,
                        uint32_t deviceCapabilityNumber,
                        VideoCaptureCapability& capability) override;

  int32_t GetBestMatchedCapability(const char* deviceUniqueIdUTF8,
                                   const VideoCaptureCapability& requested,
                                   VideoCaptureCapability& resulting) override;
  int32_t GetOrientation(const char* deviceUniqueIdUTF8,
                         VideoRotation& orientation) override;

 protected:
  /* Initialize this object*/

  virtual int32_t Init() = 0;
  /*
   * Fills the member variable _captureCapabilities with capabilities for the
   * given device name.
   */
  virtual int32_t CreateCapabilityMap(const char* deviceUniqueIdUTF8)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(_apiLock) = 0;

 protected:
  // Data members
  typedef std::vector<VideoCaptureCapability> VideoCaptureCapabilities;
  VideoCaptureCapabilities _captureCapabilities RTC_GUARDED_BY(_apiLock);
  Mutex _apiLock;
  char* _lastUsedDeviceName RTC_GUARDED_BY(_apiLock);
  uint32_t _lastUsedDeviceNameLength RTC_GUARDED_BY(_apiLock);
};
}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_DEVICE_INFO_IMPL_H_
