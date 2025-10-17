/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#include "modules/video_coding/utility/bandwidth_quality_scaler.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "api/video/video_adaptation_reason.h"
#include "api/video_codecs/video_encoder.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/exp_filter.h"
#include "rtc_base/time_utils.h"
#include "rtc_base/weak_ptr.h"

namespace webrtc {

namespace {

constexpr int kDefaultMaxWindowSizeMs = 5000;
constexpr float kHigherMaxBitrateTolerationFactor = 0.95;
constexpr float kLowerMinBitrateTolerationFactor = 0.8;
}  // namespace

BandwidthQualityScaler::BandwidthQualityScaler(
    BandwidthQualityScalerUsageHandlerInterface* handler)
    : handler_(handler),
      encoded_bitrate_(kDefaultMaxWindowSizeMs, RateStatistics::kBpsScale),
      weak_ptr_factory_(this) {
  RTC_DCHECK_RUN_ON(&task_checker_);
  RTC_DCHECK(handler_ != nullptr);

  StartCheckForBitrate();
}

BandwidthQualityScaler::~BandwidthQualityScaler() {
  RTC_DCHECK_RUN_ON(&task_checker_);
}

void BandwidthQualityScaler::StartCheckForBitrate() {
  RTC_DCHECK_RUN_ON(&task_checker_);
  TaskQueueBase::Current()->PostDelayedTask(
      [this_weak_ptr = weak_ptr_factory_.GetWeakPtr(), this] {
        if (!this_weak_ptr) {
          // The caller BandwidthQualityScaler has been deleted.
          return;
        }
        RTC_DCHECK_RUN_ON(&task_checker_);
        switch (CheckBitrate()) {
          case BandwidthQualityScaler::CheckBitrateResult::kHighBitRate: {
            handler_->OnReportUsageBandwidthHigh();
            last_frame_size_pixels_.reset();
            break;
          }
          case BandwidthQualityScaler::CheckBitrateResult::kLowBitRate: {
            handler_->OnReportUsageBandwidthLow();
            last_frame_size_pixels_.reset();
            break;
          }
          case BandwidthQualityScaler::CheckBitrateResult::kNormalBitrate: {
            break;
          }
          case BandwidthQualityScaler::CheckBitrateResult::
              kInsufficientSamples: {
            break;
          }
        }
        StartCheckForBitrate();
      },
      kBitrateStateUpdateInterval);
}

void BandwidthQualityScaler::ReportEncodeInfo(int frame_size_bytes,
                                              int64_t time_sent_in_ms,
                                              uint32_t encoded_width,
                                              uint32_t encoded_height) {
  RTC_DCHECK_RUN_ON(&task_checker_);
  last_time_sent_in_ms_ = time_sent_in_ms;
  last_frame_size_pixels_ = encoded_width * encoded_height;
  encoded_bitrate_.Update(frame_size_bytes, time_sent_in_ms);
}

void BandwidthQualityScaler::SetResolutionBitrateLimits(
    const std::vector<VideoEncoder::ResolutionBitrateLimits>&
        resolution_bitrate_limits) {
  if (resolution_bitrate_limits.empty()) {
    resolution_bitrate_limits_ = EncoderInfoSettings::
        GetDefaultSinglecastBitrateLimitsWhenQpIsUntrusted();
  } else {
    resolution_bitrate_limits_ = resolution_bitrate_limits;
  }
}

BandwidthQualityScaler::CheckBitrateResult
BandwidthQualityScaler::CheckBitrate() {
  RTC_DCHECK_RUN_ON(&task_checker_);
  if (!last_frame_size_pixels_.has_value() ||
      !last_time_sent_in_ms_.has_value()) {
    return BandwidthQualityScaler::CheckBitrateResult::kInsufficientSamples;
  }

  std::optional<int64_t> current_bitrate_bps =
      encoded_bitrate_.Rate(last_time_sent_in_ms_.value());
  if (!current_bitrate_bps.has_value()) {
    // We can't get a valid bitrate due to not enough data points.
    return BandwidthQualityScaler::CheckBitrateResult::kInsufficientSamples;
  }
  std::optional<VideoEncoder::ResolutionBitrateLimits> suitable_bitrate_limit =
      EncoderInfoSettings::
          GetSinglecastBitrateLimitForResolutionWhenQpIsUntrusted(
              last_frame_size_pixels_, resolution_bitrate_limits_);

  if (!suitable_bitrate_limit.has_value()) {
    return BandwidthQualityScaler::CheckBitrateResult::kInsufficientSamples;
  }

  // Multiply by toleration factor to solve the frequent adaptation due to
  // critical value.
  if (current_bitrate_bps > suitable_bitrate_limit->max_bitrate_bps *
                                kHigherMaxBitrateTolerationFactor) {
    return BandwidthQualityScaler::CheckBitrateResult::kLowBitRate;
  } else if (current_bitrate_bps <
             suitable_bitrate_limit->min_start_bitrate_bps *
                 kLowerMinBitrateTolerationFactor) {
    return BandwidthQualityScaler::CheckBitrateResult::kHighBitRate;
  }
  return BandwidthQualityScaler::CheckBitrateResult::kNormalBitrate;
}

BandwidthQualityScalerUsageHandlerInterface::
    ~BandwidthQualityScalerUsageHandlerInterface() {}
}  // namespace webrtc
