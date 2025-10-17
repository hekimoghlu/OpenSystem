/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
#include "rtc_base/experiments/quality_scaler_settings.h"

#include "api/field_trials_view.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {
const int kMinFrames = 10;
const double kMinScaleFactor = 0.01;
}  // namespace

QualityScalerSettings::QualityScalerSettings(
    const FieldTrialsView& field_trials)
    : sampling_period_ms_("sampling_period_ms"),
      average_qp_window_("average_qp_window"),
      min_frames_("min_frames"),
      initial_scale_factor_("initial_scale_factor"),
      scale_factor_("scale_factor"),
      initial_bitrate_interval_ms_("initial_bitrate_interval_ms"),
      initial_bitrate_factor_("initial_bitrate_factor") {
  ParseFieldTrial({&sampling_period_ms_, &average_qp_window_, &min_frames_,
                   &initial_scale_factor_, &scale_factor_,
                   &initial_bitrate_interval_ms_, &initial_bitrate_factor_},
                  field_trials.Lookup("WebRTC-Video-QualityScalerSettings"));
}

std::optional<int> QualityScalerSettings::SamplingPeriodMs() const {
  if (sampling_period_ms_ && sampling_period_ms_.Value() <= 0) {
    RTC_LOG(LS_WARNING) << "Unsupported sampling_period_ms value, ignored.";
    return std::nullopt;
  }
  return sampling_period_ms_.GetOptional();
}

std::optional<int> QualityScalerSettings::AverageQpWindow() const {
  if (average_qp_window_ && average_qp_window_.Value() <= 0) {
    RTC_LOG(LS_WARNING) << "Unsupported average_qp_window value, ignored.";
    return std::nullopt;
  }
  return average_qp_window_.GetOptional();
}

std::optional<int> QualityScalerSettings::MinFrames() const {
  if (min_frames_ && min_frames_.Value() < kMinFrames) {
    RTC_LOG(LS_WARNING) << "Unsupported min_frames value, ignored.";
    return std::nullopt;
  }
  return min_frames_.GetOptional();
}

std::optional<double> QualityScalerSettings::InitialScaleFactor() const {
  if (initial_scale_factor_ &&
      initial_scale_factor_.Value() < kMinScaleFactor) {
    RTC_LOG(LS_WARNING) << "Unsupported initial_scale_factor value, ignored.";
    return std::nullopt;
  }
  return initial_scale_factor_.GetOptional();
}

std::optional<double> QualityScalerSettings::ScaleFactor() const {
  if (scale_factor_ && scale_factor_.Value() < kMinScaleFactor) {
    RTC_LOG(LS_WARNING) << "Unsupported scale_factor value, ignored.";
    return std::nullopt;
  }
  return scale_factor_.GetOptional();
}

std::optional<int> QualityScalerSettings::InitialBitrateIntervalMs() const {
  if (initial_bitrate_interval_ms_ &&
      initial_bitrate_interval_ms_.Value() < 0) {
    RTC_LOG(LS_WARNING) << "Unsupported bitrate_interval value, ignored.";
    return std::nullopt;
  }
  return initial_bitrate_interval_ms_.GetOptional();
}

std::optional<double> QualityScalerSettings::InitialBitrateFactor() const {
  if (initial_bitrate_factor_ &&
      initial_bitrate_factor_.Value() < kMinScaleFactor) {
    RTC_LOG(LS_WARNING) << "Unsupported initial_bitrate_factor value, ignored.";
    return std::nullopt;
  }
  return initial_bitrate_factor_.GetOptional();
}

}  // namespace webrtc
