/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_MEDIA_STREAM_H_
#define SDK_ANDROID_SRC_JNI_PC_MEDIA_STREAM_H_

#include <jni.h>

#include <memory>

#include "api/media_stream_interface.h"
#include "pc/media_stream_observer.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

class JavaMediaStream {
 public:
  explicit JavaMediaStream(
      JNIEnv* env,
      rtc::scoped_refptr<MediaStreamInterface> media_stream);
  ~JavaMediaStream();

  const ScopedJavaGlobalRef<jobject>& j_media_stream() {
    return j_media_stream_;
  }

 private:
  void OnAudioTrackAddedToStream(AudioTrackInterface* track,
                                 MediaStreamInterface* stream);
  void OnVideoTrackAddedToStream(VideoTrackInterface* track,
                                 MediaStreamInterface* stream);
  void OnAudioTrackRemovedFromStream(AudioTrackInterface* track,
                                     MediaStreamInterface* stream);
  void OnVideoTrackRemovedFromStream(VideoTrackInterface* track,
                                     MediaStreamInterface* stream);

  ScopedJavaGlobalRef<jobject> j_media_stream_;
  std::unique_ptr<MediaStreamObserver> observer_;
};

jclass GetMediaStreamClass(JNIEnv* env);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_MEDIA_STREAM_H_
