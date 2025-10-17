/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
// This file contain convenience functions and classes for JNI.
// Before using any of the methods, InitGlobalJniVariables must be called.

#ifndef SDK_ANDROID_SRC_JNI_JNI_HELPERS_H_
#define SDK_ANDROID_SRC_JNI_JNI_HELPERS_H_

#include <jni.h>

#include <string>

#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"
#include "sdk/android/src/jni/jvm.h"

// Convenience macro defining JNI-accessible methods in the org.webrtc package.
// Eliminates unnecessary boilerplate and line-wraps, reducing visual clutter.
#if defined(WEBRTC_ARCH_X86)
// Dalvik JIT generated code doesn't guarantee 16-byte stack alignment on
// x86 - use force_align_arg_pointer to realign the stack at the JNI
// boundary. crbug.com/655248
#define JNI_FUNCTION_DECLARATION(rettype, name, ...)                    \
  __attribute__((force_align_arg_pointer)) extern "C" JNIEXPORT rettype \
      JNICALL Java_org_webrtc_##name(__VA_ARGS__)
#else
#define JNI_FUNCTION_DECLARATION(rettype, name, ...) \
  extern "C" JNIEXPORT rettype JNICALL Java_org_webrtc_##name(__VA_ARGS__)
#endif

namespace webrtc {
namespace jni {

// TODO(sakal): Remove once clients have migrated.
using ::webrtc::JavaToStdMapStrings;

// Deprecated, use NativeToJavaPointer.
inline long jlongFromPointer(void* ptr) {
  return NativeToJavaPointer(ptr);
}

ScopedJavaLocalRef<jobject> NewDirectByteBuffer(JNIEnv* env,
                                                void* address,
                                                jlong capacity);

jobject NewGlobalRef(JNIEnv* jni, jobject o);

void DeleteGlobalRef(JNIEnv* jni, jobject o);

// Scope Java local references to the lifetime of this object.  Use in all C++
// callbacks (i.e. entry points that don't originate in a Java callstack
// through a "native" method call).
class ScopedLocalRefFrame {
 public:
  explicit ScopedLocalRefFrame(JNIEnv* jni);
  ~ScopedLocalRefFrame();

 private:
  JNIEnv* jni_;
};

}  // namespace jni
}  // namespace webrtc

// TODO(magjed): Remove once external clients are updated.
namespace webrtc_jni {

using webrtc::AttachCurrentThreadIfNeeded;
using webrtc::jni::InitGlobalJniVariables;

}  // namespace webrtc_jni

#endif  // SDK_ANDROID_SRC_JNI_JNI_HELPERS_H_
