/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_SIMULCAST_UTILITY_H_
#define MODULES_VIDEO_CODING_UTILITY_SIMULCAST_UTILITY_H_

#include <stdint.h>

#include "api/video_codecs/video_codec.h"
#include "rtc_base/system/rtc_export.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {

class RTC_EXPORT SimulcastUtility {
 public:
  static uint32_t SumStreamMaxBitrate(int streams, const VideoCodec& codec);
  static int NumberOfSimulcastStreams(const VideoCodec& codec);
  static bool ValidSimulcastParameters(const VideoCodec& codec,
                                       int num_streams);
  static int NumberOfTemporalLayers(const VideoCodec& codec, int spatial_id);
  // TODO(sprang): Remove this hack when ScreenshareLayers is gone.
  static bool IsConferenceModeScreenshare(const VideoCodec& codec);
  static bool IsConferenceModeScreenshare(
      const VideoEncoderConfig& encoder_config);
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_SIMULCAST_UTILITY_H_
