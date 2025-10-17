/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include "sdk/android/native_api/codecs/wrapper.h"

#include <memory>

#include "sdk/android/native_api/jni/scoped_java_ref.h"
#include "sdk/android/src/jni/video_codec_info.h"
#include "sdk/android/src/jni/video_decoder_factory_wrapper.h"
#include "sdk/android/src/jni/video_encoder_factory_wrapper.h"
#include "sdk/android/src/jni/video_encoder_wrapper.h"

namespace webrtc {

SdpVideoFormat JavaToNativeVideoCodecInfo(JNIEnv* jni, jobject codec_info) {
  return jni::VideoCodecInfoToSdpVideoFormat(jni,
                                             JavaParamRef<jobject>(codec_info));
}

std::unique_ptr<VideoDecoderFactory> JavaToNativeVideoDecoderFactory(
    JNIEnv* jni,
    jobject decoder_factory) {
  return std::make_unique<jni::VideoDecoderFactoryWrapper>(
      jni, JavaParamRef<jobject>(decoder_factory));
}

std::unique_ptr<VideoEncoderFactory> JavaToNativeVideoEncoderFactory(
    JNIEnv* jni,
    jobject encoder_factory) {
  return std::make_unique<jni::VideoEncoderFactoryWrapper>(
      jni, JavaParamRef<jobject>(encoder_factory));
}

std::vector<VideoEncoder::ResolutionBitrateLimits>
JavaToNativeResolutionBitrateLimits(JNIEnv* jni,
                                    const jobjectArray j_bitrate_limits_array) {
  return jni::JavaToNativeResolutionBitrateLimits(
      jni, JavaParamRef<jobjectArray>(j_bitrate_limits_array));
}

}  // namespace webrtc
