/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#ifndef SDK_ANDROID_SRC_JNI_ENCODED_IMAGE_H_
#define SDK_ANDROID_SRC_JNI_ENCODED_IMAGE_H_

#include <jni.h>

#include <vector>

#include "api/video/video_frame_type.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {

class EncodedImage;

namespace jni {

ScopedJavaLocalRef<jobject> NativeToJavaFrameType(JNIEnv* env,
                                                  VideoFrameType frame_type);
ScopedJavaLocalRef<jobject> NativeToJavaEncodedImage(JNIEnv* jni,
                                                     const EncodedImage& image);
ScopedJavaLocalRef<jobjectArray> NativeToJavaFrameTypeArray(
    JNIEnv* env,
    const std::vector<VideoFrameType>& frame_types);

EncodedImage JavaToNativeEncodedImage(JNIEnv* env,
                                      const JavaRef<jobject>& j_encoded_image);

int64_t GetJavaEncodedImageCaptureTimeNs(
    JNIEnv* jni,
    const JavaRef<jobject>& j_encoded_image);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_ENCODED_IMAGE_H_
