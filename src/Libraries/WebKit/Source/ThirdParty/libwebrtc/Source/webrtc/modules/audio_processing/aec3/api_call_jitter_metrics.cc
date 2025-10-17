/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "modules/audio_processing/aec3/api_call_jitter_metrics.h"

#include <algorithm>
#include <limits>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {
namespace {

bool TimeToReportMetrics(int frames_since_last_report) {
  constexpr int kNumFramesPerSecond = 100;
  constexpr int kReportingIntervalFrames = 10 * kNumFramesPerSecond;
  return frames_since_last_report == kReportingIntervalFrames;
}

}  // namespace

ApiCallJitterMetrics::Jitter::Jitter()
    : max_(0), min_(std::numeric_limits<int>::max()) {}

void ApiCallJitterMetrics::Jitter::Update(int num_api_calls_in_a_row) {
  min_ = std::min(min_, num_api_calls_in_a_row);
  max_ = std::max(max_, num_api_calls_in_a_row);
}

void ApiCallJitterMetrics::Jitter::Reset() {
  min_ = std::numeric_limits<int>::max();
  max_ = 0;
}

void ApiCallJitterMetrics::Reset() {
  render_jitter_.Reset();
  capture_jitter_.Reset();
  num_api_calls_in_a_row_ = 0;
  frames_since_last_report_ = 0;
  last_call_was_render_ = false;
  proper_call_observed_ = false;
}

void ApiCallJitterMetrics::ReportRenderCall() {
  if (!last_call_was_render_) {
    // If the previous call was a capture and a proper call has been observed
    // (containing both render and capture data), storing the last number of
    // capture calls into the metrics.
    if (proper_call_observed_) {
      capture_jitter_.Update(num_api_calls_in_a_row_);
    }

    // Reset the call counter to start counting render calls.
    num_api_calls_in_a_row_ = 0;
  }
  ++num_api_calls_in_a_row_;
  last_call_was_render_ = true;
}

void ApiCallJitterMetrics::ReportCaptureCall() {
  if (last_call_was_render_) {
    // If the previous call was a render and a proper call has been observed
    // (containing both render and capture data), storing the last number of
    // render calls into the metrics.
    if (proper_call_observed_) {
      render_jitter_.Update(num_api_calls_in_a_row_);
    }
    // Reset the call counter to start counting capture calls.
    num_api_calls_in_a_row_ = 0;

    // If this statement is reached, at least one render and one capture call
    // have been observed.
    proper_call_observed_ = true;
  }
  ++num_api_calls_in_a_row_;
  last_call_was_render_ = false;

  // Only report and update jitter metrics for when a proper call, containing
  // both render and capture data, has been observed.
  if (proper_call_observed_ &&
      TimeToReportMetrics(++frames_since_last_report_)) {
    // Report jitter, where the base basic unit is frames.
    constexpr int kMaxJitterToReport = 50;

    // Report max and min jitter for render and capture, in units of 20 ms.
    RTC_HISTOGRAM_COUNTS_LINEAR(
        "WebRTC.Audio.EchoCanceller.MaxRenderJitter",
        std::min(kMaxJitterToReport, render_jitter().max()), 1,
        kMaxJitterToReport, kMaxJitterToReport);
    RTC_HISTOGRAM_COUNTS_LINEAR(
        "WebRTC.Audio.EchoCanceller.MinRenderJitter",
        std::min(kMaxJitterToReport, render_jitter().min()), 1,
        kMaxJitterToReport, kMaxJitterToReport);

    RTC_HISTOGRAM_COUNTS_LINEAR(
        "WebRTC.Audio.EchoCanceller.MaxCaptureJitter",
        std::min(kMaxJitterToReport, capture_jitter().max()), 1,
        kMaxJitterToReport, kMaxJitterToReport);
    RTC_HISTOGRAM_COUNTS_LINEAR(
        "WebRTC.Audio.EchoCanceller.MinCaptureJitter",
        std::min(kMaxJitterToReport, capture_jitter().min()), 1,
        kMaxJitterToReport, kMaxJitterToReport);

    frames_since_last_report_ = 0;
    Reset();
  }
}

bool ApiCallJitterMetrics::WillReportMetricsAtNextCapture() const {
  return TimeToReportMetrics(frames_since_last_report_ + 1);
}

}  // namespace webrtc
