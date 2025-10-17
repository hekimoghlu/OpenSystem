/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#ifndef SDK_ANDROID_NATIVE_API_VIDEO_WRAPPER_H_
#define SDK_ANDROID_NATIVE_API_VIDEO_WRAPPER_H_

#include <jni.h>

#include <memory>

#include "api/media_stream_interface.h"
#include "api/video/video_frame.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {

// Creates an instance of rtc::VideoSinkInterface<VideoFrame> from Java
// VideoSink.
std::unique_ptr<rtc::VideoSinkInterface<VideoFrame>> JavaToNativeVideoSink(
    JNIEnv* jni,
    jobject video_sink);

// Creates a Java VideoFrame object from a native VideoFrame. The returned
// object has to be released by calling release.
ScopedJavaLocalRef<jobject> NativeToJavaVideoFrame(JNIEnv* jni,
                                                   const VideoFrame& frame);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_VIDEO_WRAPPER_H_
