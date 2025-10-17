/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#ifndef VIDEO_CONFIG_ENCODER_STREAM_FACTORY_H_
#define VIDEO_CONFIG_ENCODER_STREAM_FACTORY_H_

#include <string>
#include <vector>

#include "api/field_trials_view.h"
#include "api/units/data_rate.h"
#include "api/video_codecs/video_encoder.h"
#include "call/adaptation/video_source_restrictions.h"
#include "video/config/video_encoder_config.h"

namespace cricket {

class EncoderStreamFactory
    : public webrtc::VideoEncoderConfig::VideoStreamFactoryInterface {
 public:
  EncoderStreamFactory(const webrtc::VideoEncoder::EncoderInfo& encoder_info,
                       std::optional<webrtc::VideoSourceRestrictions>
                           restrictions = std::nullopt);

  std::vector<webrtc::VideoStream> CreateEncoderStreams(
      const webrtc::FieldTrialsView& trials,
      int width,
      int height,
      const webrtc::VideoEncoderConfig& encoder_config) override;

 private:
  std::vector<webrtc::VideoStream> CreateDefaultVideoStreams(
      int width,
      int height,
      const webrtc::VideoEncoderConfig& encoder_config,
      const std::optional<webrtc::DataRate>& experimental_min_bitrate) const;

  std::vector<webrtc::VideoStream>
  CreateSimulcastOrConferenceModeScreenshareStreams(
      const webrtc::FieldTrialsView& trials,
      int width,
      int height,
      const webrtc::VideoEncoderConfig& encoder_config,
      const std::optional<webrtc::DataRate>& experimental_min_bitrate) const;

  webrtc::Resolution GetLayerResolutionFromScaleResolutionDownTo(
      int in_frame_width,
      int in_frame_height,
      webrtc::Resolution scale_resolution_down_to) const;

  std::vector<webrtc::Resolution> GetStreamResolutions(
      const webrtc::FieldTrialsView& trials,
      int width,
      int height,
      const webrtc::VideoEncoderConfig& encoder_config) const;

  const int encoder_info_requested_resolution_alignment_;
  const std::optional<webrtc::VideoSourceRestrictions> restrictions_;
};

}  // namespace cricket

#endif  // VIDEO_CONFIG_ENCODER_STREAM_FACTORY_H_
