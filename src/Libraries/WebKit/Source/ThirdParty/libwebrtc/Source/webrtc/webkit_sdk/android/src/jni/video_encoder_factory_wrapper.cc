/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "sdk/android/src/jni/video_encoder_factory_wrapper.h"

#include "api/video/render_resolution.h"
#include "api/video_codecs/video_encoder.h"
#include "rtc_base/logging.h"
#include "sdk/android/generated_video_jni/VideoEncoderFactory_jni.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/video_codec_info.h"
#include "sdk/android/src/jni/video_encoder_wrapper.h"

namespace webrtc {
namespace jni {
namespace {
class VideoEncoderSelectorWrapper
    : public VideoEncoderFactory::EncoderSelectorInterface {
 public:
  VideoEncoderSelectorWrapper(JNIEnv* jni,
                              const JavaRef<jobject>& encoder_selector)
      : encoder_selector_(jni, encoder_selector) {}

  void OnCurrentEncoder(const SdpVideoFormat& format) override {
    JNIEnv* jni = AttachCurrentThreadIfNeeded();
    ScopedJavaLocalRef<jobject> j_codec_info =
        SdpVideoFormatToVideoCodecInfo(jni, format);
    Java_VideoEncoderSelector_onCurrentEncoder(jni, encoder_selector_,
                                               j_codec_info);
  }

  absl::optional<SdpVideoFormat> OnAvailableBitrate(
      const DataRate& rate) override {
    JNIEnv* jni = AttachCurrentThreadIfNeeded();
    ScopedJavaLocalRef<jobject> codec_info =
        Java_VideoEncoderSelector_onAvailableBitrate(jni, encoder_selector_,
                                                     rate.kbps<int>());
    if (codec_info.is_null()) {
      return absl::nullopt;
    }
    return VideoCodecInfoToSdpVideoFormat(jni, codec_info);
  }

  absl::optional<SdpVideoFormat> OnResolutionChange(
      const RenderResolution& resolution) override {
    JNIEnv* jni = AttachCurrentThreadIfNeeded();
    ScopedJavaLocalRef<jobject> codec_info =
        Java_VideoEncoderSelector_onResolutionChange(
            jni, encoder_selector_, resolution.Width(), resolution.Height());
    if (codec_info.is_null()) {
      return absl::nullopt;
    }
    return VideoCodecInfoToSdpVideoFormat(jni, codec_info);
  }

  absl::optional<SdpVideoFormat> OnEncoderBroken() override {
    JNIEnv* jni = AttachCurrentThreadIfNeeded();
    ScopedJavaLocalRef<jobject> codec_info =
        Java_VideoEncoderSelector_onEncoderBroken(jni, encoder_selector_);
    if (codec_info.is_null()) {
      return absl::nullopt;
    }
    return VideoCodecInfoToSdpVideoFormat(jni, codec_info);
  }

 private:
  const ScopedJavaGlobalRef<jobject> encoder_selector_;
};

}  // namespace

VideoEncoderFactoryWrapper::VideoEncoderFactoryWrapper(
    JNIEnv* jni,
    const JavaRef<jobject>& encoder_factory)
    : encoder_factory_(jni, encoder_factory) {
  const ScopedJavaLocalRef<jobjectArray> j_supported_codecs =
      Java_VideoEncoderFactory_getSupportedCodecs(jni, encoder_factory);
  supported_formats_ = JavaToNativeVector<SdpVideoFormat>(
      jni, j_supported_codecs, &VideoCodecInfoToSdpVideoFormat);
  const ScopedJavaLocalRef<jobjectArray> j_implementations =
      Java_VideoEncoderFactory_getImplementations(jni, encoder_factory);
  implementations_ = JavaToNativeVector<SdpVideoFormat>(
      jni, j_implementations, &VideoCodecInfoToSdpVideoFormat);
}
VideoEncoderFactoryWrapper::~VideoEncoderFactoryWrapper() = default;

std::unique_ptr<VideoEncoder> VideoEncoderFactoryWrapper::CreateVideoEncoder(
    const SdpVideoFormat& format) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> j_codec_info =
      SdpVideoFormatToVideoCodecInfo(jni, format);
  ScopedJavaLocalRef<jobject> encoder = Java_VideoEncoderFactory_createEncoder(
      jni, encoder_factory_, j_codec_info);
  if (!encoder.obj())
    return nullptr;
  return JavaToNativeVideoEncoder(jni, encoder);
}

std::vector<SdpVideoFormat> VideoEncoderFactoryWrapper::GetSupportedFormats()
    const {
  return supported_formats_;
}

std::vector<SdpVideoFormat> VideoEncoderFactoryWrapper::GetImplementations()
    const {
  return implementations_;
}

std::unique_ptr<VideoEncoderFactory::EncoderSelectorInterface>
VideoEncoderFactoryWrapper::GetEncoderSelector() const {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> selector =
      Java_VideoEncoderFactory_getEncoderSelector(jni, encoder_factory_);
  if (selector.is_null()) {
    return nullptr;
  }

  return std::make_unique<VideoEncoderSelectorWrapper>(jni, selector);
}

}  // namespace jni
}  // namespace webrtc
