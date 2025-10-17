/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#ifndef SDK_ANDROID_NATIVE_API_VIDEO_VIDEO_SOURCE_H_
#define SDK_ANDROID_NATIVE_API_VIDEO_VIDEO_SOURCE_H_

#include <jni.h>

#include "api/media_stream_interface.h"
#include "rtc_base/thread.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {

// Interface for class that implements VideoTrackSourceInterface and provides a
// Java object that can be used to feed frames to the source.
class JavaVideoTrackSourceInterface : public VideoTrackSourceInterface {
 public:
  // Returns CapturerObserver object that can be used to feed frames to the
  // video source.
  virtual ScopedJavaLocalRef<jobject> GetJavaVideoCapturerObserver(
      JNIEnv* env) = 0;
};

// Creates an instance of JavaVideoTrackSourceInterface,
rtc::scoped_refptr<JavaVideoTrackSourceInterface> CreateJavaVideoSource(
    JNIEnv* env,
    rtc::Thread* signaling_thread,
    bool is_screencast,
    bool align_timestamps);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_VIDEO_VIDEO_SOURCE_H_
