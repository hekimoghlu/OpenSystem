/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#include <jni.h>

#include "api/media_stream_interface.h"
#include "sdk/android/generated_video_jni/VideoTrack_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/video_sink.h"

namespace webrtc {
namespace jni {

static void JNI_VideoTrack_AddSink(JNIEnv* jni,
                                   jlong j_native_track,
                                   jlong j_native_sink) {
  reinterpret_cast<VideoTrackInterface*>(j_native_track)
      ->AddOrUpdateSink(
          reinterpret_cast<rtc::VideoSinkInterface<VideoFrame>*>(j_native_sink),
          rtc::VideoSinkWants());
}

static void JNI_VideoTrack_RemoveSink(JNIEnv* jni,
                                      jlong j_native_track,
                                      jlong j_native_sink) {
  reinterpret_cast<VideoTrackInterface*>(j_native_track)
      ->RemoveSink(reinterpret_cast<rtc::VideoSinkInterface<VideoFrame>*>(
          j_native_sink));
}

static jlong JNI_VideoTrack_WrapSink(JNIEnv* jni,
                                     const JavaParamRef<jobject>& sink) {
  return jlongFromPointer(new VideoSinkWrapper(jni, sink));
}

static void JNI_VideoTrack_FreeSink(JNIEnv* jni, jlong j_native_sink) {
  delete reinterpret_cast<rtc::VideoSinkInterface<VideoFrame>*>(j_native_sink);
}

}  // namespace jni
}  // namespace webrtc
