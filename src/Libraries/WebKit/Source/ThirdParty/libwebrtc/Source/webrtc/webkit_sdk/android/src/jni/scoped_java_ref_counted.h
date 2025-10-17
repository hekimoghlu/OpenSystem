/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
#ifndef SDK_ANDROID_SRC_JNI_SCOPED_JAVA_REF_COUNTED_H_
#define SDK_ANDROID_SRC_JNI_SCOPED_JAVA_REF_COUNTED_H_

#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {
namespace jni {

// Holds a reference to a java object implementing the RefCounted interface, and
// calls its release() method from the destructor.
class ScopedJavaRefCounted {
 public:
  // Takes over the caller's reference.
  static ScopedJavaRefCounted Adopt(JNIEnv* jni,
                                    const JavaRef<jobject>& j_object) {
    return ScopedJavaRefCounted(jni, j_object);
  }

  // Retains the java object for the live time of this object.
  static ScopedJavaRefCounted Retain(JNIEnv* jni,
                                     const JavaRef<jobject>& j_object);
  ScopedJavaRefCounted(ScopedJavaRefCounted&& other) = default;

  ScopedJavaRefCounted(const ScopedJavaRefCounted& other) = delete;
  ScopedJavaRefCounted& operator=(const ScopedJavaRefCounted&) = delete;

  ~ScopedJavaRefCounted();

 private:
  // Adopts reference.
  ScopedJavaRefCounted(JNIEnv* jni, const JavaRef<jobject>& j_object)
      : j_object_(jni, j_object) {}

  ScopedJavaGlobalRef<jobject> j_object_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_SCOPED_JAVA_REF_COUNTED_H_
