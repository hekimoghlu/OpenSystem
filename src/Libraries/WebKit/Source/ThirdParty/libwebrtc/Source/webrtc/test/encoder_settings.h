/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
#ifndef TEST_ENCODER_SETTINGS_H_
#define TEST_ENCODER_SETTINGS_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "call/video_receive_stream.h"
#include "call/video_send_stream.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {
namespace test {

class DefaultVideoStreamFactory
    : public VideoEncoderConfig::VideoStreamFactoryInterface {
 public:
  DefaultVideoStreamFactory();

  static const size_t kMaxNumberOfStreams = 3;
  // Defined as {150000, 450000, 1500000};
  static const int kMaxBitratePerStream[];
  // Defined as {50000, 200000, 700000};
  static const int kDefaultMinBitratePerStream[];

 private:
  std::vector<VideoStream> CreateEncoderStreams(
      const FieldTrialsView& field_trials,
      int frame_width,
      int frame_height,
      const webrtc::VideoEncoderConfig& encoder_config) override;
};

// Creates `encoder_config.number_of_streams` VideoStreams where index
// `encoder_config.number_of_streams -1` have width = `width`, height =
// `height`. The total max bitrate of all VideoStreams is
// `encoder_config.max_bitrate_bps`.
std::vector<VideoStream> CreateVideoStreams(
    int width,
    int height,
    const webrtc::VideoEncoderConfig& encoder_config);

void FillEncoderConfiguration(VideoCodecType codec_type,
                              size_t num_streams,
                              VideoEncoderConfig* configuration);

VideoReceiveStreamInterface::Decoder CreateMatchingDecoder(
    int payload_type,
    const std::string& payload_name);

VideoReceiveStreamInterface::Decoder CreateMatchingDecoder(
    const VideoSendStream::Config& config);
}  // namespace test
}  // namespace webrtc

#endif  // TEST_ENCODER_SETTINGS_H_
