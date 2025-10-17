/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_BLOCK_PROCESSOR_METRICS_H_
#define MODULES_AUDIO_PROCESSING_AEC3_BLOCK_PROCESSOR_METRICS_H_

namespace webrtc {

// Handles the reporting of metrics for the block_processor.
class BlockProcessorMetrics {
 public:
  BlockProcessorMetrics() = default;

  BlockProcessorMetrics(const BlockProcessorMetrics&) = delete;
  BlockProcessorMetrics& operator=(const BlockProcessorMetrics&) = delete;

  // Updates the metric with new capture data.
  void UpdateCapture(bool underrun);

  // Updates the metric with new render data.
  void UpdateRender(bool overrun);

  // Returns true if the metrics have just been reported, otherwise false.
  bool MetricsReported() { return metrics_reported_; }

 private:
  // Resets the metrics.
  void ResetMetrics();

  int capture_block_counter_ = 0;
  bool metrics_reported_ = false;
  int render_buffer_underruns_ = 0;
  int render_buffer_overruns_ = 0;
  int buffer_render_calls_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_BLOCK_PROCESSOR_METRICS_H_
