/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#include "modules/audio_processing/aec3/render_delay_controller_metrics.h"

#include <algorithm>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {

namespace {

enum class DelayReliabilityCategory {
  kNone,
  kPoor,
  kMedium,
  kGood,
  kExcellent,
  kNumCategories
};
enum class DelayChangesCategory {
  kNone,
  kFew,
  kSeveral,
  kMany,
  kConstant,
  kNumCategories
};

}  // namespace

RenderDelayControllerMetrics::RenderDelayControllerMetrics() = default;

void RenderDelayControllerMetrics::Update(
    std::optional<size_t> delay_samples,
    std::optional<size_t> buffer_delay_blocks,
    ClockdriftDetector::Level clockdrift) {
  ++call_counter_;

  if (!initial_update) {
    size_t delay_blocks;
    if (delay_samples) {
      ++reliable_delay_estimate_counter_;
      // Add an offset by 1 (metric is halved before reporting) to reserve 0 for
      // absent delay.
      delay_blocks = (*delay_samples) / kBlockSize + 2;
    } else {
      delay_blocks = 0;
    }

    if (delay_blocks != delay_blocks_) {
      ++delay_change_counter_;
      delay_blocks_ = delay_blocks;
    }

  } else if (++initial_call_counter_ == 5 * kNumBlocksPerSecond) {
    initial_update = false;
  }

  if (call_counter_ == kMetricsReportingIntervalBlocks) {
    int value_to_report = static_cast<int>(delay_blocks_);
    // Divide by 2 to compress metric range.
    value_to_report = std::min(124, value_to_report >> 1);
    RTC_HISTOGRAM_COUNTS_LINEAR("WebRTC.Audio.EchoCanceller.EchoPathDelay",
                                value_to_report, 0, 124, 125);

    // Divide by 2 to compress metric range.
    // Offset by 1 to reserve 0 for absent delay.
    value_to_report = buffer_delay_blocks ? (*buffer_delay_blocks + 2) >> 1 : 0;
    value_to_report = std::min(124, value_to_report);
    RTC_HISTOGRAM_COUNTS_LINEAR("WebRTC.Audio.EchoCanceller.BufferDelay",
                                value_to_report, 0, 124, 125);

    DelayReliabilityCategory delay_reliability;
    if (reliable_delay_estimate_counter_ == 0) {
      delay_reliability = DelayReliabilityCategory::kNone;
    } else if (reliable_delay_estimate_counter_ > (call_counter_ >> 1)) {
      delay_reliability = DelayReliabilityCategory::kExcellent;
    } else if (reliable_delay_estimate_counter_ > 100) {
      delay_reliability = DelayReliabilityCategory::kGood;
    } else if (reliable_delay_estimate_counter_ > 10) {
      delay_reliability = DelayReliabilityCategory::kMedium;
    } else {
      delay_reliability = DelayReliabilityCategory::kPoor;
    }
    RTC_HISTOGRAM_ENUMERATION(
        "WebRTC.Audio.EchoCanceller.ReliableDelayEstimates",
        static_cast<int>(delay_reliability),
        static_cast<int>(DelayReliabilityCategory::kNumCategories));

    DelayChangesCategory delay_changes;
    if (delay_change_counter_ == 0) {
      delay_changes = DelayChangesCategory::kNone;
    } else if (delay_change_counter_ > 10) {
      delay_changes = DelayChangesCategory::kConstant;
    } else if (delay_change_counter_ > 5) {
      delay_changes = DelayChangesCategory::kMany;
    } else if (delay_change_counter_ > 2) {
      delay_changes = DelayChangesCategory::kSeveral;
    } else {
      delay_changes = DelayChangesCategory::kFew;
    }
    RTC_HISTOGRAM_ENUMERATION(
        "WebRTC.Audio.EchoCanceller.DelayChanges",
        static_cast<int>(delay_changes),
        static_cast<int>(DelayChangesCategory::kNumCategories));

    RTC_HISTOGRAM_ENUMERATION(
        "WebRTC.Audio.EchoCanceller.Clockdrift", static_cast<int>(clockdrift),
        static_cast<int>(ClockdriftDetector::Level::kNumCategories));

    call_counter_ = 0;
    ResetMetrics();
  }
}

void RenderDelayControllerMetrics::ResetMetrics() {
  delay_change_counter_ = 0;
  reliable_delay_estimate_counter_ = 0;
}

}  // namespace webrtc
