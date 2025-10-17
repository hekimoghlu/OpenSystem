/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include "api/video_codecs/video_encoder_software_fallback_wrapper.h"
#include "sdk/android/generated_video_jni/VideoEncoderFallback_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/video_encoder_wrapper.h"

namespace webrtc {
namespace jni {

static jlong JNI_VideoEncoderFallback_CreateEncoder(
    JNIEnv* jni,
    const JavaParamRef<jobject>& j_fallback_encoder,
    const JavaParamRef<jobject>& j_primary_encoder) {
  std::unique_ptr<VideoEncoder> fallback_encoder =
      JavaToNativeVideoEncoder(jni, j_fallback_encoder);
  std::unique_ptr<VideoEncoder> primary_encoder =
      JavaToNativeVideoEncoder(jni, j_primary_encoder);

  VideoEncoder* nativeWrapper =
      CreateVideoEncoderSoftwareFallbackWrapper(std::move(fallback_encoder),
                                                std::move(primary_encoder))
          .release();

  return jlongFromPointer(nativeWrapper);
}

}  // namespace jni
}  // namespace webrtc
