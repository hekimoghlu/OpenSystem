/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#include "sdk/android/native_api/jni/class_loader.h"

#include <algorithm>
#include <string>

#include "rtc_base/checks.h"
#include "sdk/android/generated_native_api_jni/WebRtcClassLoader_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

// Abort the process if `jni` has a Java exception pending. This macros uses the
// comma operator to execute ExceptionDescribe and ExceptionClear ignoring their
// return values and sending "" to the error stream.
#define CHECK_EXCEPTION(jni)        \
  RTC_CHECK(!jni->ExceptionCheck()) \
      << (jni->ExceptionDescribe(), jni->ExceptionClear(), "")

namespace webrtc {

namespace {

class ClassLoader {
 public:
  explicit ClassLoader(JNIEnv* env)
      : class_loader_(jni::Java_WebRtcClassLoader_getClassLoader(env)) {
    class_loader_class_ = reinterpret_cast<jclass>(
        env->NewGlobalRef(env->FindClass("java/lang/ClassLoader")));
    CHECK_EXCEPTION(env);
    load_class_method_ =
        env->GetMethodID(class_loader_class_, "loadClass",
                         "(Ljava/lang/String;)Ljava/lang/Class;");
    CHECK_EXCEPTION(env);
  }

  ScopedJavaLocalRef<jclass> FindClass(JNIEnv* env, const char* c_name) {
    // ClassLoader.loadClass expects a classname with components separated by
    // dots instead of the slashes that JNIEnv::FindClass expects.
    std::string name(c_name);
    std::replace(name.begin(), name.end(), '/', '.');
    ScopedJavaLocalRef<jstring> j_name = NativeToJavaString(env, name);
    const jclass clazz = static_cast<jclass>(env->CallObjectMethod(
        class_loader_.obj(), load_class_method_, j_name.obj()));
    CHECK_EXCEPTION(env);
    return ScopedJavaLocalRef<jclass>(env, clazz);
  }

 private:
  ScopedJavaGlobalRef<jobject> class_loader_;
  jclass class_loader_class_;
  jmethodID load_class_method_;
};

static ClassLoader* g_class_loader = nullptr;

}  // namespace

void InitClassLoader(JNIEnv* env) {
  RTC_CHECK(g_class_loader == nullptr);
  g_class_loader = new ClassLoader(env);
}

ScopedJavaLocalRef<jclass> GetClass(JNIEnv* env, const char* name) {
  // The class loader will be null in the JNI code called from the ClassLoader
  // ctor when we are bootstrapping ourself.
  return (g_class_loader == nullptr)
             ? ScopedJavaLocalRef<jclass>(env, env->FindClass(name))
             : g_class_loader->FindClass(env, name);
}

}  // namespace webrtc
