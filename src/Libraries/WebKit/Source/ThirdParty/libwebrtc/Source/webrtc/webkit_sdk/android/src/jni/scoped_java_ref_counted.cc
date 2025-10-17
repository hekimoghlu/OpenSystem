/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "sdk/android/src/jni/scoped_java_ref_counted.h"

#include "sdk/android/generated_base_jni/RefCounted_jni.h"

namespace webrtc {
namespace jni {

// static
ScopedJavaRefCounted ScopedJavaRefCounted::Retain(
    JNIEnv* jni,
    const JavaRef<jobject>& j_object) {
  Java_RefCounted_retain(jni, j_object);
  CHECK_EXCEPTION(jni)
      << "Unexpected java exception from java JavaRefCounted.retain()";
  return Adopt(jni, j_object);
}

ScopedJavaRefCounted::~ScopedJavaRefCounted() {
  if (!j_object_.is_null()) {
    JNIEnv* jni = AttachCurrentThreadIfNeeded();
    Java_RefCounted_release(jni, j_object_);
    CHECK_EXCEPTION(jni)
        << "Unexpected java exception from java RefCounted.release()";
  }
}

}  // namespace jni
}  // namespace webrtc
