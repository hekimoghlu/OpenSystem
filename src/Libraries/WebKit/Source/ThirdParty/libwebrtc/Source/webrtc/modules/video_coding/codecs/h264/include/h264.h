/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_H_
#define MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/video_codecs/h264_profile_level_id.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/codecs/h264/include/h264_globals.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Creates an H264 SdpVideoFormat entry with specified paramters.
RTC_EXPORT SdpVideoFormat
CreateH264Format(H264Profile profile,
                 H264Level level,
                 const std::string& packetization_mode,
                 bool add_scalability_modes = false);

// Set to disable the H.264 encoder/decoder implementations that are provided if
// `rtc_use_h264` build flag is true (if false, this function does nothing).
// This function should only be called before or during WebRTC initialization
// and is not thread-safe.
RTC_EXPORT void DisableRtcUseH264();

// Returns a vector with all supported internal H264 encode profiles that we can
// negotiate in SDP, in order of preference.
std::vector<SdpVideoFormat> SupportedH264Codecs(
    bool add_scalability_modes = false);

// Returns a vector with all supported internal H264 decode profiles that we can
// negotiate in SDP, in order of preference. This will be available for receive
// only connections.
std::vector<SdpVideoFormat> SupportedH264DecoderCodecs();

class RTC_EXPORT H264Encoder {
 public:
  // If H.264 is supported (any implementation).
  static bool IsSupported();
  static bool SupportsScalabilityMode(ScalabilityMode scalability_mode);
};

struct H264EncoderSettings {
  // Use factory function rather than constructor to allow to create
  // `H264EncoderSettings` with designated initializers.
  static H264EncoderSettings Parse(const SdpVideoFormat& format);

  H264PacketizationMode packetization_mode =
      H264PacketizationMode::NonInterleaved;
};
absl::Nonnull<std::unique_ptr<VideoEncoder>> CreateH264Encoder(
    const Environment& env,
    H264EncoderSettings settings = {});

class RTC_EXPORT H264Decoder : public VideoDecoder {
 public:
  static std::unique_ptr<H264Decoder> Create();
  static bool IsSupported();

  ~H264Decoder() override {}
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_H264_INCLUDE_H264_H_
