/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#ifndef TEST_VIDEO_CODEC_SETTINGS_H_
#define TEST_VIDEO_CODEC_SETTINGS_H_

#include "api/video_codecs/video_encoder.h"

namespace webrtc {
namespace test {

const uint16_t kTestWidth = 352;
const uint16_t kTestHeight = 288;
const uint32_t kTestFrameRate = 30;
const unsigned int kTestMinBitrateKbps = 30;
const unsigned int kTestStartBitrateKbps = 300;
const uint8_t kTestPayloadType = 100;
const int64_t kTestTimingFramesDelayMs = 200;
const uint16_t kTestOutlierFrameSizePercent = 250;

static void CodecSettings(VideoCodecType codec_type, VideoCodec* settings) {
  *settings = {};

  settings->width = kTestWidth;
  settings->height = kTestHeight;

  settings->startBitrate = kTestStartBitrateKbps;
  settings->maxBitrate = 0;
  settings->minBitrate = kTestMinBitrateKbps;

  settings->maxFramerate = kTestFrameRate;

  settings->active = true;

  settings->qpMax = 56;  // See webrtcvideoengine.h.
  settings->numberOfSimulcastStreams = 0;

  settings->timing_frame_thresholds = {
      kTestTimingFramesDelayMs,
      kTestOutlierFrameSizePercent,
  };

  settings->codecType = codec_type;
  switch (codec_type) {
    case kVideoCodecVP8:
      *(settings->VP8()) = VideoEncoder::GetDefaultVp8Settings();
      return;
    case kVideoCodecVP9:
      *(settings->VP9()) = VideoEncoder::GetDefaultVp9Settings();
      return;
    case kVideoCodecH264:
      // TODO(brandtr): Set `qpMax` here, when the OpenH264 wrapper supports it.
      *(settings->H264()) = VideoEncoder::GetDefaultH264Settings();
      return;
    default:
      return;
  }
}
}  // namespace test
}  // namespace webrtc

#endif  // TEST_VIDEO_CODEC_SETTINGS_H_
