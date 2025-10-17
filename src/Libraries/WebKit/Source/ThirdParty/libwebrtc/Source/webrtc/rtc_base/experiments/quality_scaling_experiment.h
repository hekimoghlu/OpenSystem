/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#ifndef RTC_BASE_EXPERIMENTS_QUALITY_SCALING_EXPERIMENT_H_
#define RTC_BASE_EXPERIMENTS_QUALITY_SCALING_EXPERIMENT_H_

#include <optional>

#include "api/field_trials_view.h"
#include "api/video_codecs/video_encoder.h"

namespace webrtc {
class QualityScalingExperiment {
 public:
  struct Settings {
    int vp8_low;       // VP8: low QP threshold.
    int vp8_high;      // VP8: high QP threshold.
    int vp9_low;       // VP9: low QP threshold.
    int vp9_high;      // VP9: high QP threshold.
    int h264_low;      // H264: low QP threshold.
    int h264_high;     // H264: high QP threshold.
    int generic_low;   // Generic: low QP threshold.
    int generic_high;  // Generic: high QP threshold.
    float alpha_high;  // `alpha_` for ExpFilter used when checking high QP.
    float alpha_low;   // `alpha_` for ExpFilter used when checking low QP.
    int drop;          // >0 sets `use_all_drop_reasons` to true.
  };

  // Used by QualityScaler.
  struct Config {
    float alpha_high = 0.9995f;
    float alpha_low = 0.9999f;
    // If set, all type of dropped frames are used.
    // Otherwise only dropped frames by MediaOptimization are used.
    bool use_all_drop_reasons = false;
  };

  // Returns true if the experiment is enabled.
  static bool Enabled(const FieldTrialsView& field_trials);

  // Returns settings from field trial.
  static std::optional<Settings> ParseSettings(
      const FieldTrialsView& field_trials);

  // Returns QpThresholds for the `codec_type`.
  static std::optional<VideoEncoder::QpThresholds> GetQpThresholds(
      VideoCodecType codec_type,
      const FieldTrialsView& field_trials);

  // Returns parsed values. If the parsing fails, default values are returned.
  static Config GetConfig(const FieldTrialsView& field_trials);
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_QUALITY_SCALING_EXPERIMENT_H_
