/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#include "audio_device_module.h"

#include "rtc_base/logging.h"
#include "rtc_base/ref_counted_object.h"

#include "sdk/objc/native/src/audio/audio_device_module_ios.h"

namespace webrtc {

rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceModule() {
  RTC_LOG(INFO) << __FUNCTION__;
#if defined(WEBRTC_IOS)
  return new rtc::RefCountedObject<ios_adm::AudioDeviceModuleIOS>();
#else
  RTC_LOG(LERROR)
      << "current platform is not supported => this module will self destruct!";
  return nullptr;
#endif
}
}
