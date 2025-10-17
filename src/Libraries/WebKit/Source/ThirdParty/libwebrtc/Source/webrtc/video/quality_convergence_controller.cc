/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#include "video/quality_convergence_controller.h"

#include <algorithm>

#include "rtc_base/checks.h"
#include "rtc_base/experiments/struct_parameters_parser.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {
// TODO(https://crbug.com/328598314): Remove default values once HW encoders
// correctly report the minimum QP value. These thresholds correspond to the
// default configurations used for the software encoders.
constexpr int kVp8DefaultStaticQpThreshold = 15;
constexpr int kVp9DefaultStaticQpThreshold = 32;
constexpr int kAv1DefaultStaticQpThreshold = 60;

struct StaticDetectionConfig {
  // Overrides the static QP threshold if set to a higher value than what is
  // reported by the encoder.
  std::optional<int> static_qp_threshold_override;
  std::unique_ptr<StructParametersParser> Parser();
};

std::unique_ptr<StructParametersParser> StaticDetectionConfig::Parser() {
  // The empty comments ensures that each pair is on a separate line.
  return StructParametersParser::Create("static_qp_threshold",
                                        &static_qp_threshold_override);
}

int GetDefaultStaticQpThreshold(VideoCodecType codec,
                                const FieldTrialsView& trials) {
  StaticDetectionConfig static_config;
  int default_static_qp_threhsold = 0;
  switch (codec) {
    case kVideoCodecVP8:
      default_static_qp_threhsold = kVp8DefaultStaticQpThreshold;
      static_config.Parser()->Parse(trials.Lookup("WebRTC-QCM-Static-VP8"));
      break;
    case kVideoCodecVP9:
      default_static_qp_threhsold = kVp9DefaultStaticQpThreshold;
      static_config.Parser()->Parse(trials.Lookup("WebRTC-QCM-Static-VP9"));
      break;
    case kVideoCodecAV1:
      default_static_qp_threhsold = kAv1DefaultStaticQpThreshold;
      static_config.Parser()->Parse(trials.Lookup("WebRTC-QCM-Static-AV1"));
      break;
    case kVideoCodecGeneric:
    case kVideoCodecH264:
    case kVideoCodecH265:
      // -1 will effectively disable the static QP threshold since QP values are
      // always >= 0.
      return -1;
  }

  if (static_config.static_qp_threshold_override.has_value()) {
    RTC_LOG(LS_INFO) << "static_qp_threshold_override: "
                     << *static_config.static_qp_threshold_override;
    return *static_config.static_qp_threshold_override;
  }

  return default_static_qp_threhsold;
}
}  // namespace

void QualityConvergenceController::Initialize(int number_of_layers,
                                              std::optional<int> encoder_min_qp,
                                              VideoCodecType codec,
                                              const FieldTrialsView& trials) {
  RTC_DCHECK(sequence_checker_.IsCurrent());
  RTC_CHECK(number_of_layers > 0);
  number_of_layers_ = number_of_layers;
  convergence_monitors_.clear();

  int qp_threshold = GetDefaultStaticQpThreshold(codec, trials);
  if (encoder_min_qp.has_value()) {
    qp_threshold = std::max(qp_threshold, *encoder_min_qp);
  }

  for (int i = 0; i < number_of_layers_; ++i) {
    convergence_monitors_.push_back(
        QualityConvergenceMonitor::Create(qp_threshold, codec, trials));
  }
  initialized_ = true;
}

bool QualityConvergenceController::AddSampleAndCheckTargetQuality(
    int layer_index,
    int qp,
    bool is_refresh_frame) {
  RTC_DCHECK(sequence_checker_.IsCurrent());
  RTC_CHECK(initialized_);
  if (layer_index < 0 || layer_index >= number_of_layers_) {
    return false;
  }

  // TODO(kron): Remove temporary check that verifies that the initialization is
  // working as expected. See https://crbug.com/359410061.
  RTC_DCHECK(number_of_layers_ ==
             static_cast<int>(convergence_monitors_.size()));
  if (number_of_layers_ != static_cast<int>(convergence_monitors_.size())) {
    return false;
  }

  convergence_monitors_[layer_index]->AddSample(qp, is_refresh_frame);
  return convergence_monitors_[layer_index]->AtTargetQuality();
}

}  // namespace webrtc
