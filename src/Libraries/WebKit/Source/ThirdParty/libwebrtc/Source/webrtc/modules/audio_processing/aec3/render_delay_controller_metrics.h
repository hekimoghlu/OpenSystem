/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_METRICS_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_METRICS_H_

#include <stddef.h>

#include <optional>

#include "modules/audio_processing/aec3/clockdrift_detector.h"

namespace webrtc {

// Handles the reporting of metrics for the render delay controller.
class RenderDelayControllerMetrics {
 public:
  RenderDelayControllerMetrics();

  RenderDelayControllerMetrics(const RenderDelayControllerMetrics&) = delete;
  RenderDelayControllerMetrics& operator=(const RenderDelayControllerMetrics&) =
      delete;

  // Updates the metric with new data.
  void Update(std::optional<size_t> delay_samples,
              std::optional<size_t> buffer_delay_blocks,
              ClockdriftDetector::Level clockdrift);

 private:
  // Resets the metrics.
  void ResetMetrics();

  size_t delay_blocks_ = 0;
  int reliable_delay_estimate_counter_ = 0;
  int delay_change_counter_ = 0;
  int call_counter_ = 0;
  int initial_call_counter_ = 0;
  bool initial_update = true;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_METRICS_H_
