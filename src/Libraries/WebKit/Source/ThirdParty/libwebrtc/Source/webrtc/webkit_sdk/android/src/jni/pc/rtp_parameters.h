/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_RTP_PARAMETERS_H_
#define SDK_ANDROID_SRC_JNI_PC_RTP_PARAMETERS_H_

#include <jni.h>

#include "api/rtp_parameters.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {
namespace jni {

RtpEncodingParameters JavaToNativeRtpEncodingParameters(
    JNIEnv* env,
    const JavaRef<jobject>& j_encoding_parameters);

RtpParameters JavaToNativeRtpParameters(JNIEnv* jni,
                                        const JavaRef<jobject>& j_parameters);
ScopedJavaLocalRef<jobject> NativeToJavaRtpParameters(
    JNIEnv* jni,
    const RtpParameters& parameters);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_RTP_PARAMETERS_H_
