/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_WRAPPED_NATIVE_I420_BUFFER_H_
#define SDK_ANDROID_SRC_JNI_WRAPPED_NATIVE_I420_BUFFER_H_

#include <jni.h>

#include "api/video/video_frame_buffer.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {
namespace jni {

// This function wraps the C++ I420 buffer and returns a Java
// VideoFrame.I420Buffer as a jobject.
ScopedJavaLocalRef<jobject> WrapI420Buffer(
    JNIEnv* jni,
    const rtc::scoped_refptr<I420BufferInterface>& i420_buffer);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_WRAPPED_NATIVE_I420_BUFFER_H_
