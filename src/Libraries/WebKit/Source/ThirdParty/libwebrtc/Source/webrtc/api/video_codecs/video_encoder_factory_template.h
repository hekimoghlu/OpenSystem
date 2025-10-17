/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "api/array_view.h"
#include "api/environment/environment.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/video_coding/svc/scalability_mode_util.h"

namespace webrtc {
// The VideoEncoderFactoryTemplate supports encoders implementations given as
// template arguments.
//
// To include an encoder in the factory it requires three static members
// functions to be defined:
//
//   // Returns the supported SdpVideoFormats this encoder can produce.
//   static std::vector<SdpVideoFormat> SupportedFormats();
//
//   // Creates an encoder instance for the given format.
//   static std::unique_ptr<VideoEncoder>
//       CreateEncoder(const Environment& env,
//                     const SdpVideoFormat& format);
//
//   // Returns true if the encoder supports the given scalability mode.
//   static bool
//       IsScalabilityModeSupported(ScalabilityMode scalability_mode);
//
// Note that the order of the template arguments matter as the factory will
// query/return the first encoder implementation supporting the given
// SdpVideoFormat.
template <typename... Ts>
class VideoEncoderFactoryTemplate : public VideoEncoderFactory {
 public:
  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    return GetSupportedFormatsInternal<Ts...>();
  }

  std::unique_ptr<VideoEncoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    // We fuzzy match the specified format for both valid and not so valid
    // reasons. The valid reason is that there are many standardized codec
    // specific fmtp parameters that have not been implemented, and in those
    // cases we should not fail to instantiate an encoder just because we don't
    // recognize the parameter. The not so valid reason is that we have started
    // adding parameters completely unrelated to the SDP to the SdpVideoFormat.
    // TODO: bugs.webrtc.org/13868 - Remove FuzzyMatchSdpVideoFormat
    std::optional<SdpVideoFormat> matched =
        FuzzyMatchSdpVideoFormat(GetSupportedFormats(), format);
    return CreateInternal<Ts...>(env, matched.value_or(format));
  }

  CodecSupport QueryCodecSupport(
      const SdpVideoFormat& format,
      std::optional<std::string> scalability_mode) const override {
    return QueryCodecSupportInternal<Ts...>(format, scalability_mode);
  }

 private:
  bool IsFormatInList(
      const SdpVideoFormat& format,
      rtc::ArrayView<const SdpVideoFormat> supported_formats) const {
    return absl::c_any_of(
        supported_formats, [&](const SdpVideoFormat& supported_format) {
          return supported_format.name == format.name &&
                 supported_format.parameters == format.parameters;
        });
  }

  template <typename V>
  bool IsScalabilityModeSupported(
      const std::optional<std::string>& scalability_mode_string) const {
    if (!scalability_mode_string.has_value()) {
      return true;
    }
    std::optional<ScalabilityMode> scalability_mode =
        ScalabilityModeFromString(*scalability_mode_string);
    return scalability_mode.has_value() &&
           V::IsScalabilityModeSupported(*scalability_mode);
  }

  template <typename V, typename... Vs>
  std::vector<SdpVideoFormat> GetSupportedFormatsInternal() const {
    auto supported_formats = V::SupportedFormats();

    if constexpr (sizeof...(Vs) > 0) {
      // Supported formats may overlap between implementations, so duplicates
      // should be filtered out.
      for (const auto& other_format : GetSupportedFormatsInternal<Vs...>()) {
        if (!IsFormatInList(other_format, supported_formats)) {
          supported_formats.push_back(other_format);
        }
      }
    }

    return supported_formats;
  }

  template <typename V, typename... Vs>
  std::unique_ptr<VideoEncoder> CreateInternal(const Environment& env,
                                               const SdpVideoFormat& format) {
    if (IsFormatInList(format, V::SupportedFormats())) {
      return V::CreateEncoder(env, format);
    }

    if constexpr (sizeof...(Vs) > 0) {
      return CreateInternal<Vs...>(env, format);
    }

    return nullptr;
  }

  template <typename V, typename... Vs>
  CodecSupport QueryCodecSupportInternal(
      const SdpVideoFormat& format,
      const std::optional<std::string>& scalability_mode) const {
    if (IsFormatInList(format, V::SupportedFormats())) {
      return {.is_supported = IsScalabilityModeSupported<V>(scalability_mode)};
    }

    if constexpr (sizeof...(Vs) > 0) {
      return QueryCodecSupportInternal<Vs...>(format, scalability_mode);
    }

    return {.is_supported = false};
  }
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_FACTORY_TEMPLATE_H_
