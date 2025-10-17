/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_VIDEO_DECODER_FACTORY_WRAPPER_H_
#define SDK_ANDROID_SRC_JNI_VIDEO_DECODER_FACTORY_WRAPPER_H_

#include <jni.h>

#include "api/video_codecs/video_decoder_factory.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

// Wrapper for Java VideoDecoderFactory class. Delegates method calls through
// JNI and wraps the decoder inside VideoDecoderWrapper.
class VideoDecoderFactoryWrapper : public VideoDecoderFactory {
 public:
  VideoDecoderFactoryWrapper(JNIEnv* jni,
                             const JavaRef<jobject>& decoder_factory);
  ~VideoDecoderFactoryWrapper() override;

  std::vector<SdpVideoFormat> GetSupportedFormats() const override;
  std::unique_ptr<VideoDecoder> CreateVideoDecoder(
      const SdpVideoFormat& format) override;

 private:
  const ScopedJavaGlobalRef<jobject> decoder_factory_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_VIDEO_DECODER_FACTORY_WRAPPER_H_
