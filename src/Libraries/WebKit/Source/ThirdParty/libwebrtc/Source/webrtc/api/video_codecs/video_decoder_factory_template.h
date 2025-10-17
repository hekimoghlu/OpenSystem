/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#ifndef API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_H_
#define API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_H_

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "api/array_view.h"
#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory.h"

namespace webrtc {
// The VideoDecoderFactoryTemplate supports decoder implementations given as
// template arguments.
//
// To include a decoder in the factory it requires two static members
// functions to be defined:
//
//   // Returns the supported SdpVideoFormats this decoder can decode.
//   static std::vector<SdpVideoFormat> SupportedFormats();
//
//   // Creates a decoder instance for the given format.
//   static std::unique_ptr<VideoDecoder>
//       CreateDecoder(const Environment& env,
//                     const SdpVideoFormat& format);
//
// Note that the order of the template arguments matter as the factory will
// return the first decoder implementation supporting the given SdpVideoFormat.
template <typename... Ts>
class VideoDecoderFactoryTemplate : public VideoDecoderFactory {
 public:
  std::vector<SdpVideoFormat> GetSupportedFormats() const override {
    return GetSupportedFormatsInternal<Ts...>();
  }

  std::unique_ptr<VideoDecoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override {
    return CreateVideoDecoderInternal<Ts...>(env, format);
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
  std::unique_ptr<VideoDecoder> CreateVideoDecoderInternal(
      const Environment& env,
      const SdpVideoFormat& format) {
    if (IsFormatInList(format, V::SupportedFormats())) {
      if constexpr (std::is_invocable_r_v<std::unique_ptr<VideoDecoder>,
                                          decltype(V::CreateDecoder),
                                          const Environment&,
                                          const SdpVideoFormat&>) {
        return V::CreateDecoder(env, format);
      } else {
        return V::CreateDecoder(format);
      }
    }

    if constexpr (sizeof...(Vs) > 0) {
      return CreateVideoDecoderInternal<Vs...>(env, format);
    }

    return nullptr;
  }
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_H_
