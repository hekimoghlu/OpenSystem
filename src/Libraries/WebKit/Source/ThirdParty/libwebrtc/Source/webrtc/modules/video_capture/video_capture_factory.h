/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
// This file contains interfaces used for creating the VideoCaptureModule
// and DeviceInfo.

#ifndef MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_FACTORY_H_
#define MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_FACTORY_H_

#include "api/scoped_refptr.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_defines.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class VideoCaptureOptions;

class RTC_EXPORT VideoCaptureFactory {
 public:
  // Create a video capture module object
  // id - unique identifier of this video capture module object.
  // deviceUniqueIdUTF8 - name of the device.
  //                      Available names can be found by using GetDeviceName
  static rtc::scoped_refptr<VideoCaptureModule> Create(
      const char* deviceUniqueIdUTF8);
  static rtc::scoped_refptr<VideoCaptureModule> Create(
      VideoCaptureOptions* options,
      const char* deviceUniqueIdUTF8);

  static VideoCaptureModule::DeviceInfo* CreateDeviceInfo();
  static VideoCaptureModule::DeviceInfo* CreateDeviceInfo(
      VideoCaptureOptions* options);

 private:
  ~VideoCaptureFactory();
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_VIDEO_CAPTURE_FACTORY_H_
