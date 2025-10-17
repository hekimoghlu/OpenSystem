/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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

#include "rtc_base/ssl_adapter.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

extern "C" jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM* jvm, void* reserved) {
  jint ret = InitGlobalJniVariables(jvm);
  RTC_DCHECK_GE(ret, 0);
  if (ret < 0)
    return -1;

  RTC_CHECK(rtc::InitializeSSL()) << "Failed to InitializeSSL()";
  webrtc::InitClassLoader(GetEnv());

  return ret;
}

extern "C" void JNIEXPORT JNICALL JNI_OnUnLoad(JavaVM* jvm, void* reserved) {
  RTC_CHECK(rtc::CleanupSSL()) << "Failed to CleanupSSL()";
}

}  // namespace jni
}  // namespace webrtc
