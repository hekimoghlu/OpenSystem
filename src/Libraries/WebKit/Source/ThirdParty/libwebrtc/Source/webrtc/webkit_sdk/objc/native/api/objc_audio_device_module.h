/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef SDK_OBJC_NATIVE_API_OBJC_AUDIO_DEVICE_MODULE_H_
#define SDK_OBJC_NATIVE_API_OBJC_AUDIO_DEVICE_MODULE_H_

#import "components/audio/RTCAudioDevice.h"
#include "modules/audio_device/include/audio_device.h"

namespace webrtc {

rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceModule(
    id<RTC_OBJC_TYPE(RTCAudioDevice)> audio_device);

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_API_OBJC_AUDIO_DEVICE_MODULE_H_
