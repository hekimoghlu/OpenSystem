/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "api/video_codecs/builtin_video_decoder_factory.h"
#include "api/video_codecs/video_decoder.h"
#include "sdk/android/generated_swcodecs_jni/SoftwareVideoDecoderFactory_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/video_codec_info.h"

namespace webrtc {
namespace jni {

static jlong JNI_SoftwareVideoDecoderFactory_CreateFactory(JNIEnv* env) {
  return webrtc::NativeToJavaPointer(
      CreateBuiltinVideoDecoderFactory().release());
}

static jlong JNI_SoftwareVideoDecoderFactory_CreateDecoder(
    JNIEnv* env,
    jlong j_factory,
    const webrtc::JavaParamRef<jobject>& j_video_codec_info) {
  auto* const native_factory =
      reinterpret_cast<webrtc::VideoDecoderFactory*>(j_factory);
  const auto video_format =
      webrtc::jni::VideoCodecInfoToSdpVideoFormat(env, j_video_codec_info);

  auto decoder = native_factory->CreateVideoDecoder(video_format);
  if (decoder == nullptr) {
    return 0;
  }
  return webrtc::NativeToJavaPointer(decoder.release());
}

static webrtc::ScopedJavaLocalRef<jobject>
JNI_SoftwareVideoDecoderFactory_GetSupportedCodecs(JNIEnv* env,
                                                   jlong j_factory) {
  auto* const native_factory =
      reinterpret_cast<webrtc::VideoDecoderFactory*>(j_factory);

  return webrtc::NativeToJavaList(env, native_factory->GetSupportedFormats(),
                                  &webrtc::jni::SdpVideoFormatToVideoCodecInfo);
}

}  // namespace jni
}  // namespace webrtc
