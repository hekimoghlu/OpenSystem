/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_RTC_CERTIFICATE_H_
#define SDK_ANDROID_SRC_JNI_PC_RTC_CERTIFICATE_H_

#include "rtc_base/ref_count.h"
#include "rtc_base/rtc_certificate.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

rtc::RTCCertificatePEM JavaToNativeRTCCertificatePEM(
    JNIEnv* jni,
    const JavaRef<jobject>& j_rtc_certificate);

ScopedJavaLocalRef<jobject> NativeToJavaRTCCertificatePEM(
    JNIEnv* env,
    const rtc::RTCCertificatePEM& certificate);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_RTC_CERTIFICATE_H_
