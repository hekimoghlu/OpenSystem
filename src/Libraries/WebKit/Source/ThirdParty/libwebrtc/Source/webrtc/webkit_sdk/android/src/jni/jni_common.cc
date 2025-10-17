/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#include "rtc_base/ref_count.h"
#include "sdk/android/generated_base_jni/JniCommon_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static void JNI_JniCommon_AddRef(JNIEnv* jni,
                                 jlong j_native_ref_counted_pointer) {
  reinterpret_cast<rtc::RefCountInterface*>(j_native_ref_counted_pointer)
      ->AddRef();
}

static void JNI_JniCommon_ReleaseRef(JNIEnv* jni,
                                     jlong j_native_ref_counted_pointer) {
  reinterpret_cast<rtc::RefCountInterface*>(j_native_ref_counted_pointer)
      ->Release();
}

static ScopedJavaLocalRef<jobject> JNI_JniCommon_AllocateByteBuffer(JNIEnv* jni,
                                                                    jint size) {
  void* new_data = ::operator new(size);
  return NewDirectByteBuffer(jni, new_data, size);
}

static void JNI_JniCommon_FreeByteBuffer(
    JNIEnv* jni,
    const JavaParamRef<jobject>& byte_buffer) {
  void* data = jni->GetDirectBufferAddress(byte_buffer.obj());
  ::operator delete(data);
}

}  // namespace jni
}  // namespace webrtc
