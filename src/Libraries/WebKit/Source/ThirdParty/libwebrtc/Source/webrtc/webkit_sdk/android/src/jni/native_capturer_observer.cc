/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "sdk/android/src/jni/native_capturer_observer.h"

#include "rtc_base/logging.h"
#include "sdk/android/generated_video_jni/NativeCapturerObserver_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/android_video_track_source.h"

namespace webrtc {
namespace jni {

ScopedJavaLocalRef<jobject> CreateJavaNativeCapturerObserver(
    JNIEnv* env,
    rtc::scoped_refptr<AndroidVideoTrackSource> native_source) {
  return Java_NativeCapturerObserver_Constructor(
      env, NativeToJavaPointer(native_source.release()));
}

}  // namespace jni
}  // namespace webrtc
