/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

#include <new>
#include <string>

#include "api/scoped_refptr.h"
#include "media/base/video_common.h"
#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/video_capture/linux/video_capture_pipewire.h"
#endif
#include "modules/video_capture/linux/video_capture_v4l2.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_options.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace videocapturemodule {
rtc::scoped_refptr<VideoCaptureModule> VideoCaptureImpl::Create(
    const char* deviceUniqueId) {
  auto implementation = rtc::make_ref_counted<VideoCaptureModuleV4L2>();

  if (implementation->Init(deviceUniqueId) != 0)
    return nullptr;

  return implementation;
}

rtc::scoped_refptr<VideoCaptureModule> VideoCaptureImpl::Create(
    VideoCaptureOptions* options,
    const char* deviceUniqueId) {
#if defined(WEBRTC_USE_PIPEWIRE)
  if (options->allow_pipewire()) {
    auto implementation =
        rtc::make_ref_counted<VideoCaptureModulePipeWire>(options);

    if (implementation->Init(deviceUniqueId) == 0)
      return implementation;
  }
#endif
  if (options->allow_v4l2()) {
    auto implementation = rtc::make_ref_counted<VideoCaptureModuleV4L2>();

    if (implementation->Init(deviceUniqueId) == 0)
      return implementation;
  }
  return nullptr;
}
}  // namespace videocapturemodule
}  // namespace webrtc
