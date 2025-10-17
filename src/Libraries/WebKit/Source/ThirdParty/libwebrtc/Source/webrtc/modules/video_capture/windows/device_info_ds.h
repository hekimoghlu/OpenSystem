/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_DEVICE_INFO_DS_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_DEVICE_INFO_DS_H_

#include <dshow.h>

#include "modules/video_capture/device_info_impl.h"
#include "modules/video_capture/video_capture_impl.h"

namespace webrtc {
namespace videocapturemodule {
struct VideoCaptureCapabilityWindows : public VideoCaptureCapability {
  uint32_t directShowCapabilityIndex;
  bool supportFrameRateControl;
  VideoCaptureCapabilityWindows() {
    directShowCapabilityIndex = 0;
    supportFrameRateControl = false;
  }
};

class DeviceInfoDS : public DeviceInfoImpl {
 public:
  // Factory function.
  static DeviceInfoDS* Create();

  DeviceInfoDS();
  ~DeviceInfoDS() override;

  int32_t Init() override;
  uint32_t NumberOfDevices() override;

  /*
   * Returns the available capture devices.
   */
  int32_t GetDeviceName(uint32_t deviceNumber,
                        char* deviceNameUTF8,
                        uint32_t deviceNameLength,
                        char* deviceUniqueIdUTF8,
                        uint32_t deviceUniqueIdUTF8Length,
                        char* productUniqueIdUTF8,
                        uint32_t productUniqueIdUTF8Length) override;

  /*
   * Display OS /capture device specific settings dialog
   */
  int32_t DisplayCaptureSettingsDialogBox(const char* deviceUniqueIdUTF8,
                                          const char* dialogTitleUTF8,
                                          void* parentWindow,
                                          uint32_t positionX,
                                          uint32_t positionY) override;

  // Windows specific

  /* Gets a capture device filter
   The user of this API is responsible for releasing the filter when it not
   needed.
   */
  IBaseFilter* GetDeviceFilter(const char* deviceUniqueIdUTF8,
                               char* productUniqueIdUTF8 = NULL,
                               uint32_t productUniqueIdUTF8Length = 0);

  int32_t GetWindowsCapability(
      int32_t capabilityIndex,
      VideoCaptureCapabilityWindows& windowsCapability);

  static void GetProductId(const char* devicePath,
                           char* productUniqueIdUTF8,
                           uint32_t productUniqueIdUTF8Length);

 protected:
  int32_t GetDeviceInfo(uint32_t deviceNumber,
                        char* deviceNameUTF8,
                        uint32_t deviceNameLength,
                        char* deviceUniqueIdUTF8,
                        uint32_t deviceUniqueIdUTF8Length,
                        char* productUniqueIdUTF8,
                        uint32_t productUniqueIdUTF8Length);

  int32_t CreateCapabilityMap(const char* deviceUniqueIdUTF8) override
      RTC_EXCLUSIVE_LOCKS_REQUIRED(_apiLock);

 private:
  ICreateDevEnum* _dsDevEnum;
  IEnumMoniker* _dsMonikerDevEnum;
  bool _CoUninitializeIsRequired;
  std::vector<VideoCaptureCapabilityWindows> _captureCapabilitiesWindows;
};
}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_DEVICE_INFO_DS_H_
