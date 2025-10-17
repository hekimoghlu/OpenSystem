/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "test/encoder_settings.h"

#include <algorithm>

#include "api/scoped_refptr.h"
#include "api/video_codecs/sdp_video_format.h"
#include "call/rtp_config.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

const size_t DefaultVideoStreamFactory::kMaxNumberOfStreams;
const int DefaultVideoStreamFactory::kMaxBitratePerStream[] = {150000, 450000,
                                                               1500000};
const int DefaultVideoStreamFactory::kDefaultMinBitratePerStream[] = {
    30000, 200000, 700000};

// static
std::vector<VideoStream> CreateVideoStreams(
    int width,
    int height,
    const webrtc::VideoEncoderConfig& encoder_config) {
  RTC_DCHECK(encoder_config.number_of_streams <=
             DefaultVideoStreamFactory::kMaxNumberOfStreams);

  std::vector<VideoStream> stream_settings(encoder_config.number_of_streams);

  int bitrate_left_bps = 0;
  if (encoder_config.max_bitrate_bps > 0) {
    bitrate_left_bps = encoder_config.max_bitrate_bps;
  } else {
    for (size_t stream_num = 0; stream_num < encoder_config.number_of_streams;
         ++stream_num) {
      bitrate_left_bps +=
          DefaultVideoStreamFactory::kMaxBitratePerStream[stream_num];
    }
  }

  for (size_t i = 0; i < encoder_config.number_of_streams; ++i) {
    stream_settings[i].width =
        (i + 1) * width / encoder_config.number_of_streams;
    stream_settings[i].height =
        (i + 1) * height / encoder_config.number_of_streams;
    stream_settings[i].max_framerate = 30;
    stream_settings[i].max_qp = 56;
    stream_settings[i].min_bitrate_bps =
        DefaultVideoStreamFactory::kDefaultMinBitratePerStream[i];

    // Use configured values instead of default values if set.
    const VideoStream stream = (i < encoder_config.simulcast_layers.size())
                                   ? encoder_config.simulcast_layers[i]
                                   : VideoStream();

    int max_bitrate_bps =
        stream.max_bitrate_bps > 0
            ? stream.max_bitrate_bps
            : DefaultVideoStreamFactory::kMaxBitratePerStream[i];
    max_bitrate_bps = std::min(bitrate_left_bps, max_bitrate_bps);

    int target_bitrate_bps = stream.target_bitrate_bps > 0
                                 ? stream.target_bitrate_bps
                                 : max_bitrate_bps;
    target_bitrate_bps = std::min(max_bitrate_bps, target_bitrate_bps);

    if (stream.min_bitrate_bps > 0) {
      RTC_DCHECK_LE(stream.min_bitrate_bps, target_bitrate_bps);
      stream_settings[i].min_bitrate_bps = stream.min_bitrate_bps;
    }
    if (stream.max_framerate > 0) {
      stream_settings[i].max_framerate = stream.max_framerate;
    }
    if (stream.num_temporal_layers) {
      RTC_DCHECK_GE(*stream.num_temporal_layers, 1);
      stream_settings[i].num_temporal_layers = stream.num_temporal_layers;
    }
    if (stream.scale_resolution_down_by >= 1.0) {
      stream_settings[i].width = width / stream.scale_resolution_down_by;
      stream_settings[i].height = height / stream.scale_resolution_down_by;
    }
    stream_settings[i].scalability_mode = stream.scalability_mode;
    stream_settings[i].target_bitrate_bps = target_bitrate_bps;
    stream_settings[i].max_bitrate_bps = max_bitrate_bps;
    stream_settings[i].active =
        encoder_config.number_of_streams == 1 || stream.active;

    bitrate_left_bps -= stream_settings[i].target_bitrate_bps;
  }

  stream_settings[encoder_config.number_of_streams - 1].max_bitrate_bps +=
      bitrate_left_bps;
  stream_settings[0].bitrate_priority = encoder_config.bitrate_priority;

  return stream_settings;
}

DefaultVideoStreamFactory::DefaultVideoStreamFactory() {}

std::vector<VideoStream> DefaultVideoStreamFactory::CreateEncoderStreams(
    const FieldTrialsView& /*field_trials*/,
    int frame_width,
    int frame_height,
    const webrtc::VideoEncoderConfig& encoder_config) {
  return CreateVideoStreams(frame_width, frame_height, encoder_config);
}

void FillEncoderConfiguration(VideoCodecType codec_type,
                              size_t num_streams,
                              VideoEncoderConfig* configuration) {
  RTC_DCHECK_LE(num_streams, DefaultVideoStreamFactory::kMaxNumberOfStreams);

  configuration->codec_type = codec_type;
  configuration->number_of_streams = num_streams;
  configuration->video_stream_factory =
      rtc::make_ref_counted<DefaultVideoStreamFactory>();
  configuration->max_bitrate_bps = 0;
  configuration->frame_drop_enabled = true;
  configuration->simulcast_layers = std::vector<VideoStream>(num_streams);
  for (size_t i = 0; i < num_streams; ++i) {
    configuration->max_bitrate_bps +=
        DefaultVideoStreamFactory::kMaxBitratePerStream[i];
  }
}

VideoReceiveStreamInterface::Decoder CreateMatchingDecoder(
    int payload_type,
    const std::string& payload_name) {
  VideoReceiveStreamInterface::Decoder decoder;
  decoder.payload_type = payload_type;
  decoder.video_format = SdpVideoFormat(payload_name);
  return decoder;
}

VideoReceiveStreamInterface::Decoder CreateMatchingDecoder(
    const VideoSendStream::Config& config) {
  return CreateMatchingDecoder(config.rtp.payload_type,
                               config.rtp.payload_name);
}

}  // namespace test
}  // namespace webrtc
