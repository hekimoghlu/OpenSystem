/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#ifndef CALL_ADAPTATION_ENCODER_SETTINGS_H_
#define CALL_ADAPTATION_ENCODER_SETTINGS_H_

#include <optional>

#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {

// Information about an encoder available when reconfiguring the encoder.
class EncoderSettings {
 public:
  EncoderSettings(VideoEncoder::EncoderInfo encoder_info,
                  VideoEncoderConfig encoder_config,
                  VideoCodec video_codec);
  EncoderSettings(const EncoderSettings& other);
  EncoderSettings& operator=(const EncoderSettings& other);

  // Encoder capabilities, implementation info, etc.
  const VideoEncoder::EncoderInfo& encoder_info() const;
  // Configuration parameters, ultimately coming from the API and negotiation.
  const VideoEncoderConfig& encoder_config() const;
  // Lower level config, heavily based on the VideoEncoderConfig.
  const VideoCodec& video_codec() const;

 private:
  VideoEncoder::EncoderInfo encoder_info_;
  VideoEncoderConfig encoder_config_;
  VideoCodec video_codec_;
};

VideoCodecType GetVideoCodecTypeOrGeneric(
    const std::optional<EncoderSettings>& settings);

}  // namespace webrtc

#endif  // CALL_ADAPTATION_ENCODER_SETTINGS_H_
