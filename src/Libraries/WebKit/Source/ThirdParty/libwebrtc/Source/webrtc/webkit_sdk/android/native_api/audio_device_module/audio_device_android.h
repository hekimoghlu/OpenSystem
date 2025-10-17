/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#ifndef SDK_ANDROID_NATIVE_API_AUDIO_DEVICE_MODULE_AUDIO_DEVICE_ANDROID_H_
#define SDK_ANDROID_NATIVE_API_AUDIO_DEVICE_MODULE_AUDIO_DEVICE_ANDROID_H_

#include <jni.h>

#include "modules/audio_device/include/audio_device.h"

namespace webrtc {

#if defined(WEBRTC_AUDIO_DEVICE_INCLUDE_ANDROID_AAUDIO)
rtc::scoped_refptr<AudioDeviceModule> CreateAAudioAudioDeviceModule(
    JNIEnv* env,
    jobject application_context);
#endif

rtc::scoped_refptr<AudioDeviceModule> CreateJavaAudioDeviceModule(
    JNIEnv* env,
    jobject application_context);

rtc::scoped_refptr<AudioDeviceModule> CreateOpenSLESAudioDeviceModule(
    JNIEnv* env,
    jobject application_context);

rtc::scoped_refptr<AudioDeviceModule>
CreateJavaInputAndOpenSLESOutputAudioDeviceModule(
    JNIEnv* env,
    jobject application_context);

rtc::scoped_refptr<AudioDeviceModule>
CreateJavaInputAndAAudioOutputAudioDeviceModule(
    JNIEnv* env,
    jobject application_context);

rtc::scoped_refptr<AudioDeviceModule> CreateAndroidAudioDeviceModule(
    AudioDeviceModule::AudioLayer audio_layer);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_AUDIO_DEVICE_MODULE_AUDIO_DEVICE_ANDROID_H_
