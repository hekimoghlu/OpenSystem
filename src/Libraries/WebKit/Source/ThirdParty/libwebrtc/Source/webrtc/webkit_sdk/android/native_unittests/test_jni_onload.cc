/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#include <jni.h>
#undef JNIEXPORT
#define JNIEXPORT __attribute__((visibility("default")))

#include "rtc_base/checks.h"
#include "sdk/android/native_api/base/init.h"
#include "sdk/android/native_api/jni/java_types.h"

// This is called by the VM when the shared library is first loaded.
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  webrtc::InitAndroid(vm);
  return JNI_VERSION_1_4;
}
