/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#include "modules/congestion_controller/goog_cc/bitrate_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "api/field_trials_view.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/checks.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

namespace {
constexpr int kInitialRateWindowMs = 500;
constexpr int kRateWindowMs = 150;
constexpr int kMinRateWindowMs = 150;
constexpr int kMaxRateWindowMs = 1000;

const char kBweThroughputWindowConfig[] = "WebRTC-BweThroughputWindowConfig";

}  // namespace

BitrateEstimator::BitrateEstimator(const FieldTrialsView* key_value_config)
    : sum_(0),
      initial_window_ms_("initial_window_ms",
                         kInitialRateWindowMs,
                         kMinRateWindowMs,
                         kMaxRateWindowMs),
      noninitial_window_ms_("window_ms",
                            kRateWindowMs,
                            kMinRateWindowMs,
                            kMaxRateWindowMs),
      uncertainty_scale_("scale", 10.0),
      uncertainty_scale_in_alr_("scale_alr", uncertainty_scale_),
      small_sample_uncertainty_scale_("scale_small", uncertainty_scale_),
      small_sample_threshold_("small_thresh", DataSize::Zero()),
      uncertainty_symmetry_cap_("symmetry_cap", DataRate::Zero()),
      estimate_floor_("floor", DataRate::Zero()),
      current_window_ms_(0),
      prev_time_ms_(-1),
      bitrate_estimate_kbps_(-1.0f),
      bitrate_estimate_var_(50.0f) {
  // E.g WebRTC-BweThroughputWindowConfig/initial_window_ms:350,window_ms:250/
  ParseFieldTrial(
      {&initial_window_ms_, &noninitial_window_ms_, &uncertainty_scale_,
       &uncertainty_scale_in_alr_, &small_sample_uncertainty_scale_,
       &small_sample_threshold_, &uncertainty_symmetry_cap_, &estimate_floor_},
      key_value_config->Lookup(kBweThroughputWindowConfig));
}

BitrateEstimator::~BitrateEstimator() = default;

void BitrateEstimator::Update(Timestamp at_time, DataSize amount, bool in_alr) {
  int rate_window_ms = noninitial_window_ms_.Get();
  // We use a larger window at the beginning to get a more stable sample that
  // we can use to initialize the estimate.
  if (bitrate_estimate_kbps_ < 0.f)
    rate_window_ms = initial_window_ms_.Get();
  bool is_small_sample = false;
  float bitrate_sample_kbps = UpdateWindow(at_time.ms(), amount.bytes(),
                                           rate_window_ms, &is_small_sample);
  if (bitrate_sample_kbps < 0.0f)
    return;
  if (bitrate_estimate_kbps_ < 0.0f) {
    // This is the very first sample we get. Use it to initialize the estimate.
    bitrate_estimate_kbps_ = bitrate_sample_kbps;
    return;
  }
  // Optionally use higher uncertainty for very small samples to avoid dropping
  // estimate and for samples obtained in ALR.
  float scale = uncertainty_scale_;
  if (is_small_sample && bitrate_sample_kbps < bitrate_estimate_kbps_) {
    scale = small_sample_uncertainty_scale_;
  } else if (in_alr && bitrate_sample_kbps < bitrate_estimate_kbps_) {
    // Optionally use higher uncertainty for samples obtained during ALR.
    scale = uncertainty_scale_in_alr_;
  }
  // Define the sample uncertainty as a function of how far away it is from the
  // current estimate. With low values of uncertainty_symmetry_cap_ we add more
  // uncertainty to increases than to decreases. For higher values we approach
  // symmetry.
  float sample_uncertainty =
      scale * std::abs(bitrate_estimate_kbps_ - bitrate_sample_kbps) /
      (bitrate_estimate_kbps_ +
       std::min(bitrate_sample_kbps,
                uncertainty_symmetry_cap_.Get().kbps<float>()));

  float sample_var = sample_uncertainty * sample_uncertainty;
  // Update a bayesian estimate of the rate, weighting it lower if the sample
  // uncertainty is large.
  // The bitrate estimate uncertainty is increased with each update to model
  // that the bitrate changes over time.
  float pred_bitrate_estimate_var = bitrate_estimate_var_ + 5.f;
  bitrate_estimate_kbps_ = (sample_var * bitrate_estimate_kbps_ +
                            pred_bitrate_estimate_var * bitrate_sample_kbps) /
                           (sample_var + pred_bitrate_estimate_var);
  bitrate_estimate_kbps_ =
      std::max(bitrate_estimate_kbps_, estimate_floor_.Get().kbps<float>());
  bitrate_estimate_var_ = sample_var * pred_bitrate_estimate_var /
                          (sample_var + pred_bitrate_estimate_var);
}

float BitrateEstimator::UpdateWindow(int64_t now_ms,
                                     int bytes,
                                     int rate_window_ms,
                                     bool* is_small_sample) {
  RTC_DCHECK(is_small_sample != nullptr);
  // Reset if time moves backwards.
  if (now_ms < prev_time_ms_) {
    prev_time_ms_ = -1;
    sum_ = 0;
    current_window_ms_ = 0;
  }
  if (prev_time_ms_ >= 0) {
    current_window_ms_ += now_ms - prev_time_ms_;
    // Reset if nothing has been received for more than a full window.
    if (now_ms - prev_time_ms_ > rate_window_ms) {
      sum_ = 0;
      current_window_ms_ %= rate_window_ms;
    }
  }
  prev_time_ms_ = now_ms;
  float bitrate_sample = -1.0f;
  if (current_window_ms_ >= rate_window_ms) {
    *is_small_sample = sum_ < small_sample_threshold_->bytes();
    bitrate_sample = 8.0f * sum_ / static_cast<float>(rate_window_ms);
    current_window_ms_ -= rate_window_ms;
    sum_ = 0;
  }
  sum_ += bytes;
  return bitrate_sample;
}

std::optional<DataRate> BitrateEstimator::bitrate() const {
  if (bitrate_estimate_kbps_ < 0.f)
    return std::nullopt;
  return DataRate::KilobitsPerSec(bitrate_estimate_kbps_);
}

std::optional<DataRate> BitrateEstimator::PeekRate() const {
  if (current_window_ms_ > 0)
    return DataSize::Bytes(sum_) / TimeDelta::Millis(current_window_ms_);
  return std::nullopt;
}

void BitrateEstimator::ExpectFastRateChange() {
  // By setting the bitrate-estimate variance to a higher value we allow the
  // bitrate to change fast for the next few samples.
  bitrate_estimate_var_ += 200;
}

}  // namespace webrtc
