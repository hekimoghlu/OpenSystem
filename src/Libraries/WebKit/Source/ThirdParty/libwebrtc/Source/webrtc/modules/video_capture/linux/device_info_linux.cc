/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
// v4l includes
#include <linux/videodev2.h>

#include <vector>

#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/video_capture/linux/device_info_pipewire.h"
#endif
#include "modules/video_capture/linux/device_info_v4l2.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_defines.h"
#include "modules/video_capture/video_capture_impl.h"
#include "modules/video_capture/video_capture_options.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace videocapturemodule {
VideoCaptureModule::DeviceInfo* VideoCaptureImpl::CreateDeviceInfo() {
  return new videocapturemodule::DeviceInfoV4l2();
}

VideoCaptureModule::DeviceInfo* VideoCaptureImpl::CreateDeviceInfo(
    VideoCaptureOptions* options) {
#if defined(WEBRTC_USE_PIPEWIRE)
  if (options->allow_pipewire()) {
    return new videocapturemodule::DeviceInfoPipeWire(options);
  }
#endif
  if (options->allow_v4l2())
    return new videocapturemodule::DeviceInfoV4l2();

  return nullptr;
}
}  // namespace videocapturemodule
}  // namespace webrtc
