/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#include "api/video/i420_buffer.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/video_frame.h"
#include "sdk/android/src/jni/wrapped_native_i420_buffer.h"

namespace webrtc {
namespace jni {

JNI_FUNCTION_DECLARATION(jint,
                         VideoFrameBufferTest_nativeGetBufferType,
                         JNIEnv* jni,
                         jclass,
                         jobject video_frame_buffer) {
  const JavaParamRef<jobject> j_video_frame_buffer(video_frame_buffer);
  rtc::scoped_refptr<VideoFrameBuffer> buffer =
      JavaToNativeFrameBuffer(jni, j_video_frame_buffer);
  return static_cast<jint>(buffer->type());
}

JNI_FUNCTION_DECLARATION(jobject,
                         VideoFrameBufferTest_nativeGetNativeI420Buffer,
                         JNIEnv* jni,
                         jclass,
                         jobject i420_buffer) {
  const JavaParamRef<jobject> j_i420_buffer(i420_buffer);
  rtc::scoped_refptr<VideoFrameBuffer> buffer =
      JavaToNativeFrameBuffer(jni, j_i420_buffer);
  const I420BufferInterface* inputBuffer = buffer->GetI420();
  RTC_DCHECK(inputBuffer != nullptr);
  rtc::scoped_refptr<I420Buffer> outputBuffer = I420Buffer::Copy(*inputBuffer);
  return WrapI420Buffer(jni, outputBuffer).Release();
}

}  // namespace jni
}  // namespace webrtc
