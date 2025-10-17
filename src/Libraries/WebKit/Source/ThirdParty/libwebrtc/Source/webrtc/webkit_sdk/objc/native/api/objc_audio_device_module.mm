/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#include "objc_audio_device_module.h"

#include "api/make_ref_counted.h"
#include "rtc_base/logging.h"

#include "sdk/objc/native/src/objc_audio_device.h"

namespace webrtc {

rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceModule(
    id<RTC_OBJC_TYPE(RTCAudioDevice)> audio_device) {
  RTC_DLOG(LS_INFO) << __FUNCTION__;
  return rtc::make_ref_counted<objc_adm::ObjCAudioDeviceModule>(audio_device);
}

}  // namespace webrtc
