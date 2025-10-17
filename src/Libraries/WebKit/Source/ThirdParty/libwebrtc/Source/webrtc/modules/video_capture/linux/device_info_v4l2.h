/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef MODULES_VIDEO_CAPTURE_LINUX_DEVICE_INFO_V4L2_H_
#define MODULES_VIDEO_CAPTURE_LINUX_DEVICE_INFO_V4L2_H_

#include <stdint.h>

#include "modules/video_capture/device_info_impl.h"

namespace webrtc {
namespace videocapturemodule {
class DeviceInfoV4l2 : public DeviceInfoImpl {
 public:
  DeviceInfoV4l2();
  ~DeviceInfoV4l2() override;
  uint32_t NumberOfDevices() override;
  int32_t GetDeviceName(uint32_t deviceNumber,
                        char* deviceNameUTF8,
                        uint32_t deviceNameLength,
                        char* deviceUniqueIdUTF8,
                        uint32_t deviceUniqueIdUTF8Length,
                        char* productUniqueIdUTF8 = 0,
                        uint32_t productUniqueIdUTF8Length = 0) override;
  /*
   * Fills the membervariable _captureCapabilities with capabilites for the
   * given device name.
   */
  int32_t CreateCapabilityMap(const char* deviceUniqueIdUTF8) override
      RTC_EXCLUSIVE_LOCKS_REQUIRED(_apiLock);
  int32_t DisplayCaptureSettingsDialogBox(const char* /*deviceUniqueIdUTF8*/,
                                          const char* /*dialogTitleUTF8*/,
                                          void* /*parentWindow*/,
                                          uint32_t /*positionX*/,
                                          uint32_t /*positionY*/) override;
  int32_t FillCapabilities(int fd) RTC_EXCLUSIVE_LOCKS_REQUIRED(_apiLock);
  int32_t Init() override;

 private:
  bool IsDeviceNameMatches(const char* name, const char* deviceUniqueIdUTF8);
};
}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_LINUX_DEVICE_INFO_V4L2_H_
