/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include "sdk/android/src/jni/video_codec_info.h"

#include "sdk/android/generated_video_jni/VideoCodecInfo_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

SdpVideoFormat VideoCodecInfoToSdpVideoFormat(JNIEnv* jni,
                                              const JavaRef<jobject>& j_info) {
  return SdpVideoFormat(
      JavaToNativeString(jni, Java_VideoCodecInfo_getName(jni, j_info)),
      JavaToNativeStringMap(jni, Java_VideoCodecInfo_getParams(jni, j_info)));
}

ScopedJavaLocalRef<jobject> SdpVideoFormatToVideoCodecInfo(
    JNIEnv* jni,
    const SdpVideoFormat& format) {
  ScopedJavaLocalRef<jobject> j_params =
      NativeToJavaStringMap(jni, format.parameters);
  return Java_VideoCodecInfo_Constructor(
      jni, NativeToJavaString(jni, format.name), j_params);
}

}  // namespace jni
}  // namespace webrtc
