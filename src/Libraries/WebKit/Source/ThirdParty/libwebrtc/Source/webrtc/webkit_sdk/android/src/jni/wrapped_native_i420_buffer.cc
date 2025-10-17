/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "sdk/android/src/jni/wrapped_native_i420_buffer.h"

#include "sdk/android/generated_video_jni/WrappedNativeI420Buffer_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

// TODO(magjed): Write a test for this function.
ScopedJavaLocalRef<jobject> WrapI420Buffer(
    JNIEnv* jni,
    const rtc::scoped_refptr<I420BufferInterface>& i420_buffer) {
  ScopedJavaLocalRef<jobject> y_buffer =
      NewDirectByteBuffer(jni, const_cast<uint8_t*>(i420_buffer->DataY()),
                          i420_buffer->StrideY() * i420_buffer->height());
  ScopedJavaLocalRef<jobject> u_buffer =
      NewDirectByteBuffer(jni, const_cast<uint8_t*>(i420_buffer->DataU()),
                          i420_buffer->StrideU() * i420_buffer->ChromaHeight());
  ScopedJavaLocalRef<jobject> v_buffer =
      NewDirectByteBuffer(jni, const_cast<uint8_t*>(i420_buffer->DataV()),
                          i420_buffer->StrideV() * i420_buffer->ChromaHeight());

  return Java_WrappedNativeI420Buffer_Constructor(
      jni, i420_buffer->width(), i420_buffer->height(), y_buffer,
      i420_buffer->StrideY(), u_buffer, i420_buffer->StrideU(), v_buffer,
      i420_buffer->StrideV(), jlongFromPointer(i420_buffer.get()));
}

}  // namespace jni
}  // namespace webrtc
