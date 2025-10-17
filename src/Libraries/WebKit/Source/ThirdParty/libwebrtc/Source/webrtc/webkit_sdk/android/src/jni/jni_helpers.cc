/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "sdk/android/src/jni/jni_helpers.h"

#include <vector>

#include "sdk/android/native_api/jni/java_types.h"

namespace webrtc {
namespace jni {

ScopedJavaLocalRef<jobject> NewDirectByteBuffer(JNIEnv* env,
                                                void* address,
                                                jlong capacity) {
  ScopedJavaLocalRef<jobject> buffer(
      env, env->NewDirectByteBuffer(address, capacity));
  CHECK_EXCEPTION(env) << "error NewDirectByteBuffer";
  return buffer;
}

jobject NewGlobalRef(JNIEnv* jni, jobject o) {
  jobject ret = jni->NewGlobalRef(o);
  CHECK_EXCEPTION(jni) << "error during NewGlobalRef";
  RTC_CHECK(ret);
  return ret;
}

void DeleteGlobalRef(JNIEnv* jni, jobject o) {
  jni->DeleteGlobalRef(o);
  CHECK_EXCEPTION(jni) << "error during DeleteGlobalRef";
}

// Scope Java local references to the lifetime of this object.  Use in all C++
// callbacks (i.e. entry points that don't originate in a Java callstack
// through a "native" method call).
ScopedLocalRefFrame::ScopedLocalRefFrame(JNIEnv* jni) : jni_(jni) {
  RTC_CHECK(!jni_->PushLocalFrame(0)) << "Failed to PushLocalFrame";
}
ScopedLocalRefFrame::~ScopedLocalRefFrame() {
  jni_->PopLocalFrame(nullptr);
}

}  // namespace jni
}  // namespace webrtc
