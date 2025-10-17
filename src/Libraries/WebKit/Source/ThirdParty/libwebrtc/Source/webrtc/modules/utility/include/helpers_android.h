/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifndef MODULES_UTILITY_INCLUDE_HELPERS_ANDROID_H_
#define MODULES_UTILITY_INCLUDE_HELPERS_ANDROID_H_

#include <jni.h>

#include <string>

#include "rtc_base/system/arch.h"

// Abort the process if `jni` has a Java exception pending.
// TODO(henrika): merge with CHECK_JNI_EXCEPTION() in jni_helpers.h.
#define CHECK_EXCEPTION(jni)        \
  RTC_CHECK(!jni->ExceptionCheck()) \
      << (jni->ExceptionDescribe(), jni->ExceptionClear(), "")

#if defined(WEBRTC_ARCH_X86)
// Dalvik JIT generated code doesn't guarantee 16-byte stack alignment on
// x86 - use force_align_arg_pointer to realign the stack at the JNI
// boundary. bugs.webrtc.org/9050
#define JNI_FUNCTION_ALIGN __attribute__((force_align_arg_pointer))
#else
#define JNI_FUNCTION_ALIGN
#endif

namespace webrtc {

// Return a |JNIEnv*| usable on this thread or NULL if this thread is detached.
JNIEnv* GetEnv(JavaVM* jvm);

// Return a `jlong` that will correctly convert back to `ptr`.  This is needed
// because the alternative (of silently passing a 32-bit pointer to a vararg
// function expecting a 64-bit param) picks up garbage in the high 32 bits.
jlong PointerTojlong(void* ptr);

// JNIEnv-helper methods that wraps the API which uses the JNI interface
// pointer (JNIEnv*). It allows us to RTC_CHECK success and that no Java
// exception is thrown while calling the method.
jmethodID GetMethodID(JNIEnv* jni,
                      jclass c,
                      const char* name,
                      const char* signature);

jmethodID GetStaticMethodID(JNIEnv* jni,
                            jclass c,
                            const char* name,
                            const char* signature);

jclass FindClass(JNIEnv* jni, const char* name);

jobject NewGlobalRef(JNIEnv* jni, jobject o);

void DeleteGlobalRef(JNIEnv* jni, jobject o);

// Attach thread to JVM if necessary and detach at scope end if originally
// attached.
class AttachThreadScoped {
 public:
  explicit AttachThreadScoped(JavaVM* jvm);
  ~AttachThreadScoped();
  JNIEnv* env();

 private:
  bool attached_;
  JavaVM* jvm_;
  JNIEnv* env_;
};

}  // namespace webrtc

#endif  // MODULES_UTILITY_INCLUDE_HELPERS_ANDROID_H_
