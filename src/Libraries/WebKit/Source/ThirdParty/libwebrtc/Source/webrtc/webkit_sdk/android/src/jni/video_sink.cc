/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#include "sdk/android/src/jni/video_sink.h"

#include "sdk/android/generated_video_jni/VideoSink_jni.h"
#include "sdk/android/src/jni/video_frame.h"

namespace webrtc {
namespace jni {

VideoSinkWrapper::VideoSinkWrapper(JNIEnv* jni, const JavaRef<jobject>& j_sink)
    : j_sink_(jni, j_sink) {}

VideoSinkWrapper::~VideoSinkWrapper() {}

void VideoSinkWrapper::OnFrame(const VideoFrame& frame) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> j_frame = NativeToJavaVideoFrame(jni, frame);
  Java_VideoSink_onFrame(jni, j_sink_, j_frame);
  ReleaseJavaVideoFrame(jni, j_frame);
}

}  // namespace jni
}  // namespace webrtc
