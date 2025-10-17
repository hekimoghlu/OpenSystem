/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "call/adaptation/encoder_settings.h"

#include <utility>

namespace webrtc {

EncoderSettings::EncoderSettings(VideoEncoder::EncoderInfo encoder_info,
                                 VideoEncoderConfig encoder_config,
                                 VideoCodec video_codec)
    : encoder_info_(std::move(encoder_info)),
      encoder_config_(std::move(encoder_config)),
      video_codec_(std::move(video_codec)) {}

EncoderSettings::EncoderSettings(const EncoderSettings& other)
    : encoder_info_(other.encoder_info_),
      encoder_config_(other.encoder_config_.Copy()),
      video_codec_(other.video_codec_) {}

EncoderSettings& EncoderSettings::operator=(const EncoderSettings& other) {
  encoder_info_ = other.encoder_info_;
  encoder_config_ = other.encoder_config_.Copy();
  video_codec_ = other.video_codec_;
  return *this;
}

const VideoEncoder::EncoderInfo& EncoderSettings::encoder_info() const {
  return encoder_info_;
}

const VideoEncoderConfig& EncoderSettings::encoder_config() const {
  return encoder_config_;
}

const VideoCodec& EncoderSettings::video_codec() const {
  return video_codec_;
}

VideoCodecType GetVideoCodecTypeOrGeneric(
    const std::optional<EncoderSettings>& settings) {
  return settings.has_value() ? settings->encoder_config().codec_type
                              : kVideoCodecGeneric;
}

}  // namespace webrtc
