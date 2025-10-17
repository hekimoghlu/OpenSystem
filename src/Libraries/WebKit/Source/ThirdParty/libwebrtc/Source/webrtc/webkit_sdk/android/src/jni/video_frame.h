/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_VIDEO_FRAME_H_
#define SDK_ANDROID_SRC_JNI_VIDEO_FRAME_H_

#include <jni.h>

#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

rtc::scoped_refptr<VideoFrameBuffer> JavaToNativeFrameBuffer(
    JNIEnv* jni,
    const JavaRef<jobject>& j_video_frame_buffer);

VideoFrame JavaToNativeFrame(JNIEnv* jni,
                             const JavaRef<jobject>& j_video_frame,
                             uint32_t timestamp_rtp);

// NOTE: Returns a new video frame that has to be released by calling
// ReleaseJavaVideoFrame.
ScopedJavaLocalRef<jobject> NativeToJavaVideoFrame(JNIEnv* jni,
                                                   const VideoFrame& frame);
void ReleaseJavaVideoFrame(JNIEnv* jni, const JavaRef<jobject>& j_video_frame);

int64_t GetJavaVideoFrameTimestampNs(JNIEnv* jni,
                                     const JavaRef<jobject>& j_video_frame);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_VIDEO_FRAME_H_
