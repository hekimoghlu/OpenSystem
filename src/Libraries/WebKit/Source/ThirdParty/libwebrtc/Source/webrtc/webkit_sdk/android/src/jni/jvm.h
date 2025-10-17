/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_JVM_H_
#define SDK_ANDROID_SRC_JNI_JVM_H_

#include <jni.h>

namespace webrtc {
namespace jni {

jint InitGlobalJniVariables(JavaVM* jvm);

// Return a |JNIEnv*| usable on this thread or NULL if this thread is detached.
JNIEnv* GetEnv();

JavaVM* GetJVM();

// Return a |JNIEnv*| usable on this thread.  Attaches to `g_jvm` if necessary.
JNIEnv* AttachCurrentThreadIfNeeded();

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_JVM_H_
