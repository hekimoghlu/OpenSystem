/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#include "api/scoped_refptr.h"
#include "modules/video_capture/windows/video_capture_ds.h"

namespace webrtc {
namespace videocapturemodule {

// static
VideoCaptureModule::DeviceInfo* VideoCaptureImpl::CreateDeviceInfo() {
  // TODO(tommi): Use the Media Foundation version on Vista and up.
  return DeviceInfoDS::Create();
}

rtc::scoped_refptr<VideoCaptureModule> VideoCaptureImpl::Create(
    const char* device_id) {
  if (device_id == nullptr)
    return nullptr;

  // TODO(tommi): Use Media Foundation implementation for Vista and up.
  auto capture = rtc::make_ref_counted<VideoCaptureDS>();
  if (capture->Init(device_id) != 0) {
    return nullptr;
  }

  return capture;
}

}  // namespace videocapturemodule
}  // namespace webrtc
