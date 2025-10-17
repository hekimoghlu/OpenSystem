/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_VIDEO_SINK_H_
#define SDK_ANDROID_SRC_JNI_VIDEO_SINK_H_

#include <jni.h>

#include "api/media_stream_interface.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

class VideoSinkWrapper : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  VideoSinkWrapper(JNIEnv* jni, const JavaRef<jobject>& j_sink);
  ~VideoSinkWrapper() override;

 private:
  void OnFrame(const VideoFrame& frame) override;

  const ScopedJavaGlobalRef<jobject> j_sink_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_VIDEO_SINK_H_
