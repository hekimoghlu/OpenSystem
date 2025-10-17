/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "sdk/android/src/jni/pc/media_stream_track.h"

#include "api/media_stream_interface.h"
#include "sdk/android/generated_peerconnection_jni/MediaStreamTrack_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

ScopedJavaLocalRef<jobject> NativeToJavaMediaType(
    JNIEnv* jni,
    cricket::MediaType media_type) {
  return Java_MediaType_fromNativeIndex(jni, media_type);
}

cricket::MediaType JavaToNativeMediaType(JNIEnv* jni,
                                         const JavaRef<jobject>& j_media_type) {
  return static_cast<cricket::MediaType>(
      Java_MediaType_getNative(jni, j_media_type));
}

static ScopedJavaLocalRef<jstring> JNI_MediaStreamTrack_GetId(JNIEnv* jni,
                                                              jlong j_p) {
  return NativeToJavaString(
      jni, reinterpret_cast<MediaStreamTrackInterface*>(j_p)->id());
}

static ScopedJavaLocalRef<jstring> JNI_MediaStreamTrack_GetKind(JNIEnv* jni,
                                                                jlong j_p) {
  return NativeToJavaString(
      jni, reinterpret_cast<MediaStreamTrackInterface*>(j_p)->kind());
}

static jboolean JNI_MediaStreamTrack_GetEnabled(JNIEnv* jni, jlong j_p) {
  return reinterpret_cast<MediaStreamTrackInterface*>(j_p)->enabled();
}

static ScopedJavaLocalRef<jobject> JNI_MediaStreamTrack_GetState(JNIEnv* jni,
                                                                 jlong j_p) {
  return Java_State_fromNativeIndex(
      jni, reinterpret_cast<MediaStreamTrackInterface*>(j_p)->state());
}

static jboolean JNI_MediaStreamTrack_SetEnabled(JNIEnv* jni,
                                                jlong j_p,
                                                jboolean enabled) {
  return reinterpret_cast<MediaStreamTrackInterface*>(j_p)->set_enabled(
      enabled);
}

}  // namespace jni
}  // namespace webrtc
