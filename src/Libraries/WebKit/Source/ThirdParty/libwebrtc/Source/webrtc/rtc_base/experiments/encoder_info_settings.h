/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#ifndef RTC_BASE_EXPERIMENTS_ENCODER_INFO_SETTINGS_H_
#define RTC_BASE_EXPERIMENTS_ENCODER_INFO_SETTINGS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/field_trials_view.h"
#include "api/video_codecs/video_encoder.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

class EncoderInfoSettings {
 public:
  virtual ~EncoderInfoSettings();

  // Bitrate limits per resolution.
  struct BitrateLimit {
    int frame_size_pixels = 0;      // The video frame size.
    int min_start_bitrate_bps = 0;  // The minimum bitrate to start encoding.
    int min_bitrate_bps = 0;        // The minimum bitrate.
    int max_bitrate_bps = 0;        // The maximum bitrate.
  };

  std::optional<uint32_t> requested_resolution_alignment() const;
  bool apply_alignment_to_all_simulcast_layers() const {
    return apply_alignment_to_all_simulcast_layers_.Get();
  }
  std::vector<VideoEncoder::ResolutionBitrateLimits> resolution_bitrate_limits()
      const {
    return resolution_bitrate_limits_;
  }

  static std::vector<VideoEncoder::ResolutionBitrateLimits>
  GetDefaultSinglecastBitrateLimits(VideoCodecType codec_type);

  static std::optional<VideoEncoder::ResolutionBitrateLimits>
  GetDefaultSinglecastBitrateLimitsForResolution(VideoCodecType codec_type,
                                                 int frame_size_pixels);

  static std::vector<VideoEncoder::ResolutionBitrateLimits>
  GetDefaultSinglecastBitrateLimitsWhenQpIsUntrusted();

  static std::optional<VideoEncoder::ResolutionBitrateLimits>
  GetSinglecastBitrateLimitForResolutionWhenQpIsUntrusted(
      std::optional<int> frame_size_pixels,
      const std::vector<VideoEncoder::ResolutionBitrateLimits>&
          resolution_bitrate_limits);

 protected:
  EncoderInfoSettings(const FieldTrialsView& field_trials,
                      absl::string_view name);

 private:
  FieldTrialOptional<uint32_t> requested_resolution_alignment_;
  FieldTrialFlag apply_alignment_to_all_simulcast_layers_;
  std::vector<VideoEncoder::ResolutionBitrateLimits> resolution_bitrate_limits_;
};

// EncoderInfo settings for SimulcastEncoderAdapter.
class SimulcastEncoderAdapterEncoderInfoSettings : public EncoderInfoSettings {
 public:
  explicit SimulcastEncoderAdapterEncoderInfoSettings(
      const FieldTrialsView& field_trials);
  ~SimulcastEncoderAdapterEncoderInfoSettings() override {}
};

// EncoderInfo settings for LibvpxVp8Encoder.
class LibvpxVp8EncoderInfoSettings : public EncoderInfoSettings {
 public:
  explicit LibvpxVp8EncoderInfoSettings(const FieldTrialsView& field_trials);
  ~LibvpxVp8EncoderInfoSettings() override {}
};

// EncoderInfo settings for LibvpxVp9Encoder.
class LibvpxVp9EncoderInfoSettings : public EncoderInfoSettings {
 public:
  explicit LibvpxVp9EncoderInfoSettings(const FieldTrialsView& field_trials);
  ~LibvpxVp9EncoderInfoSettings() override {}
};

// EncoderInfo settings for LibaomAv1Encoder.
class LibaomAv1EncoderInfoSettings : public EncoderInfoSettings {
 public:
  explicit LibaomAv1EncoderInfoSettings(const FieldTrialsView& field_trials);
  ~LibaomAv1EncoderInfoSettings() override {}
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_ENCODER_INFO_SETTINGS_H_
