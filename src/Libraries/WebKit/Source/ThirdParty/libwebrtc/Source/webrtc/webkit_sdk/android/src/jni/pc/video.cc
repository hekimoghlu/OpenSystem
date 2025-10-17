/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 25, 2025.
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
#include "sdk/android/src/jni/pc/video.h"

#include <jni.h>

#include <memory>

#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "rtc_base/logging.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/android_video_track_source.h"
#include "sdk/android/src/jni/video_decoder_factory_wrapper.h"
#include "sdk/android/src/jni/video_encoder_factory_wrapper.h"

namespace webrtc {
namespace jni {

VideoEncoderFactory* CreateVideoEncoderFactory(
    JNIEnv* jni,
    const JavaRef<jobject>& j_encoder_factory) {
  return IsNull(jni, j_encoder_factory)
             ? nullptr
             : new VideoEncoderFactoryWrapper(jni, j_encoder_factory);
}

VideoDecoderFactory* CreateVideoDecoderFactory(
    JNIEnv* jni,
    const JavaRef<jobject>& j_decoder_factory) {
  return IsNull(jni, j_decoder_factory)
             ? nullptr
             : new VideoDecoderFactoryWrapper(jni, j_decoder_factory);
}

void* CreateVideoSource(JNIEnv* env,
                        rtc::Thread* signaling_thread,
                        rtc::Thread* worker_thread,
                        jboolean is_screencast,
                        jboolean align_timestamps) {
  auto source = rtc::make_ref_counted<AndroidVideoTrackSource>(
      signaling_thread, env, is_screencast, align_timestamps);
  return source.release();
}

}  // namespace jni
}  // namespace webrtc
