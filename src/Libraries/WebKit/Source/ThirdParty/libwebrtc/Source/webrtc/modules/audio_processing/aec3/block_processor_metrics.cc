/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#include "modules/audio_processing/aec3/block_processor_metrics.h"

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {

namespace {

enum class RenderUnderrunCategory {
  kNone,
  kFew,
  kSeveral,
  kMany,
  kConstant,
  kNumCategories
};

enum class RenderOverrunCategory {
  kNone,
  kFew,
  kSeveral,
  kMany,
  kConstant,
  kNumCategories
};

}  // namespace

void BlockProcessorMetrics::UpdateCapture(bool underrun) {
  ++capture_block_counter_;
  if (underrun) {
    ++render_buffer_underruns_;
  }

  if (capture_block_counter_ == kMetricsReportingIntervalBlocks) {
    metrics_reported_ = true;

    RenderUnderrunCategory underrun_category;
    if (render_buffer_underruns_ == 0) {
      underrun_category = RenderUnderrunCategory::kNone;
    } else if (render_buffer_underruns_ > (capture_block_counter_ >> 1)) {
      underrun_category = RenderUnderrunCategory::kConstant;
    } else if (render_buffer_underruns_ > 100) {
      underrun_category = RenderUnderrunCategory::kMany;
    } else if (render_buffer_underruns_ > 10) {
      underrun_category = RenderUnderrunCategory::kSeveral;
    } else {
      underrun_category = RenderUnderrunCategory::kFew;
    }
    RTC_HISTOGRAM_ENUMERATION(
        "WebRTC.Audio.EchoCanceller.RenderUnderruns",
        static_cast<int>(underrun_category),
        static_cast<int>(RenderUnderrunCategory::kNumCategories));

    RenderOverrunCategory overrun_category;
    if (render_buffer_overruns_ == 0) {
      overrun_category = RenderOverrunCategory::kNone;
    } else if (render_buffer_overruns_ > (buffer_render_calls_ >> 1)) {
      overrun_category = RenderOverrunCategory::kConstant;
    } else if (render_buffer_overruns_ > 100) {
      overrun_category = RenderOverrunCategory::kMany;
    } else if (render_buffer_overruns_ > 10) {
      overrun_category = RenderOverrunCategory::kSeveral;
    } else {
      overrun_category = RenderOverrunCategory::kFew;
    }
    RTC_HISTOGRAM_ENUMERATION(
        "WebRTC.Audio.EchoCanceller.RenderOverruns",
        static_cast<int>(overrun_category),
        static_cast<int>(RenderOverrunCategory::kNumCategories));

    ResetMetrics();
    capture_block_counter_ = 0;
  } else {
    metrics_reported_ = false;
  }
}

void BlockProcessorMetrics::UpdateRender(bool overrun) {
  ++buffer_render_calls_;
  if (overrun) {
    ++render_buffer_overruns_;
  }
}

void BlockProcessorMetrics::ResetMetrics() {
  render_buffer_underruns_ = 0;
  render_buffer_overruns_ = 0;
  buffer_render_calls_ = 0;
}

}  // namespace webrtc
