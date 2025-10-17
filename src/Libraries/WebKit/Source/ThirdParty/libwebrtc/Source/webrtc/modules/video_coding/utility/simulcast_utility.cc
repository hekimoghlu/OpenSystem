/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#include "modules/video_coding/utility/simulcast_utility.h"

#include <algorithm>
#include <cmath>

#include "modules/video_coding/svc/scalability_mode_util.h"
#include "rtc_base/checks.h"

namespace webrtc {

uint32_t SimulcastUtility::SumStreamMaxBitrate(int streams,
                                               const VideoCodec& codec) {
  uint32_t bitrate_sum = 0;
  for (int i = 0; i < streams; ++i) {
    bitrate_sum += codec.simulcastStream[i].maxBitrate;
  }
  return bitrate_sum;
}

int SimulcastUtility::NumberOfSimulcastStreams(const VideoCodec& codec) {
  int streams =
      codec.numberOfSimulcastStreams < 1 ? 1 : codec.numberOfSimulcastStreams;
  uint32_t simulcast_max_bitrate = SumStreamMaxBitrate(streams, codec);
  if (simulcast_max_bitrate == 0) {
    streams = 1;
  }
  return streams;
}

bool SimulcastUtility::ValidSimulcastParameters(const VideoCodec& codec,
                                                int num_streams) {
  // Check resolution.
  if (codec.width != codec.simulcastStream[num_streams - 1].width ||
      codec.height != codec.simulcastStream[num_streams - 1].height) {
    return false;
  }
  for (int i = 0; i < num_streams; ++i) {
    if (codec.width * codec.simulcastStream[i].height !=
        codec.height * codec.simulcastStream[i].width) {
      return false;
    }
  }
  for (int i = 1; i < num_streams; ++i) {
    if (codec.simulcastStream[i].width < codec.simulcastStream[i - 1].width) {
      return false;
    }
  }

  // Check frame-rate.
  for (int i = 1; i < num_streams; ++i) {
    if (fabs(codec.simulcastStream[i].maxFramerate -
             codec.simulcastStream[i - 1].maxFramerate) > 1e-9) {
      return false;
    }
  }

  // Check temporal layers.
  for (int i = 0; i < num_streams - 1; ++i) {
    if (codec.simulcastStream[i].numberOfTemporalLayers !=
        codec.simulcastStream[i + 1].numberOfTemporalLayers)
      return false;
  }
  return true;
}

bool SimulcastUtility::IsConferenceModeScreenshare(const VideoCodec& codec) {
  return codec.mode == VideoCodecMode::kScreensharing &&
         codec.legacy_conference_mode &&
         (codec.codecType == kVideoCodecVP8 ||
          codec.codecType == kVideoCodecH264);
}

bool SimulcastUtility::IsConferenceModeScreenshare(
    const VideoEncoderConfig& encoder_config) {
  return encoder_config.content_type ==
             VideoEncoderConfig::ContentType::kScreen &&
         encoder_config.legacy_conference_mode &&
         (encoder_config.codec_type == webrtc::VideoCodecType::kVideoCodecVP8 ||
          encoder_config.codec_type == webrtc::VideoCodecType::kVideoCodecH264);
}

int SimulcastUtility::NumberOfTemporalLayers(const VideoCodec& codec,
                                             int spatial_id) {
  int num_temporal_layers = 0;
  if (auto scalability_mode = codec.GetScalabilityMode(); scalability_mode) {
    num_temporal_layers = ScalabilityModeToNumTemporalLayers(*scalability_mode);
  } else {
    switch (codec.codecType) {
      case kVideoCodecVP8:
        num_temporal_layers = codec.VP8().numberOfTemporalLayers;
        break;
      case kVideoCodecVP9:
        num_temporal_layers = codec.VP9().numberOfTemporalLayers;
        break;
      case kVideoCodecH264:
        num_temporal_layers = codec.H264().numberOfTemporalLayers;
        break;
      // For AV1 and H.265 we get temporal layer count from scalability mode,
      // instead of from codec-specifics.
      case kVideoCodecAV1:
      case kVideoCodecH265:
      case kVideoCodecGeneric:
        break;
    }
  }

  if (codec.numberOfSimulcastStreams > 0) {
    RTC_DCHECK_LT(spatial_id, codec.numberOfSimulcastStreams);
    num_temporal_layers =
        std::max(num_temporal_layers,
                 static_cast<int>(
                     codec.simulcastStream[spatial_id].numberOfTemporalLayers));
  }
  return std::max(1, num_temporal_layers);
}

}  // namespace webrtc
