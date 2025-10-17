/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "modules/video_capture/video_capture_factory.h"

#include "modules/video_capture/video_capture_impl.h"

namespace webrtc {

rtc::scoped_refptr<VideoCaptureModule> VideoCaptureFactory::Create(
    const char* deviceUniqueIdUTF8) {
#if defined(WEBRTC_ANDROID) || defined(WEBRTC_MAC)
  return nullptr;
#else
  return videocapturemodule::VideoCaptureImpl::Create(deviceUniqueIdUTF8);
#endif
}

rtc::scoped_refptr<VideoCaptureModule> VideoCaptureFactory::Create(
    VideoCaptureOptions* options,
    const char* deviceUniqueIdUTF8) {
// This is only implemented on pure Linux and WEBRTC_LINUX is defined for
// Android as well
#if !defined(WEBRTC_LINUX) || defined(WEBRTC_ANDROID)
  return nullptr;
#else
  return videocapturemodule::VideoCaptureImpl::Create(options,
                                                      deviceUniqueIdUTF8);
#endif
}

VideoCaptureModule::DeviceInfo* VideoCaptureFactory::CreateDeviceInfo() {
#if defined(WEBRTC_ANDROID) || defined(WEBRTC_MAC)
  return nullptr;
#else
  return videocapturemodule::VideoCaptureImpl::CreateDeviceInfo();
#endif
}

VideoCaptureModule::DeviceInfo* VideoCaptureFactory::CreateDeviceInfo(
    VideoCaptureOptions* options) {
// This is only implemented on pure Linux and WEBRTC_LINUX is defined for
// Android as well
#if !defined(WEBRTC_LINUX) || defined(WEBRTC_ANDROID)
  return nullptr;
#else
  return videocapturemodule::VideoCaptureImpl::CreateDeviceInfo(options);
#endif
}

}  // namespace webrtc
