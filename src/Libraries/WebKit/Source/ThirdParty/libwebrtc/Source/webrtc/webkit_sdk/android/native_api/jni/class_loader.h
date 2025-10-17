/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
// Android's FindClass() is tricky because the app-specific ClassLoader is not
// consulted when there is no app-specific frame on the stack (i.e. when called
// from a thread created from native C++ code). These helper functions provide a
// workaround for this.
// http://developer.android.com/training/articles/perf-jni.html#faq_FindClass

#ifndef SDK_ANDROID_NATIVE_API_JNI_CLASS_LOADER_H_
#define SDK_ANDROID_NATIVE_API_JNI_CLASS_LOADER_H_

#include <jni.h>

#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {

// This method should be called from JNI_OnLoad and before any calls to
// FindClass. This is normally called by InitAndroid.
void InitClassLoader(JNIEnv* env);

// This function is identical to JNIEnv::FindClass except that it works from any
// thread. This function loads and returns a local reference to the class with
// the given name. The name argument is a fully-qualified class name. For
// example, the fully-qualified class name for the java.lang.String class is:
// "java/lang/String". This function will be used from the JNI generated code
// and should rarely be used manually.
ScopedJavaLocalRef<jclass> GetClass(JNIEnv* env, const char* name);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_JNI_CLASS_LOADER_H_
