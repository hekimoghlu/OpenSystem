/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_API_CALL_JITTER_METRICS_H_
#define MODULES_AUDIO_PROCESSING_AEC3_API_CALL_JITTER_METRICS_H_

namespace webrtc {

// Stores data for reporting metrics on the API call jitter.
class ApiCallJitterMetrics {
 public:
  class Jitter {
   public:
    Jitter();
    void Update(int num_api_calls_in_a_row);
    void Reset();

    int min() const { return min_; }
    int max() const { return max_; }

   private:
    int max_;
    int min_;
  };

  ApiCallJitterMetrics() { Reset(); }

  // Update metrics for render API call.
  void ReportRenderCall();

  // Update and periodically report metrics for capture API call.
  void ReportCaptureCall();

  // Methods used only for testing.
  const Jitter& render_jitter() const { return render_jitter_; }
  const Jitter& capture_jitter() const { return capture_jitter_; }
  bool WillReportMetricsAtNextCapture() const;

 private:
  void Reset();

  Jitter render_jitter_;
  Jitter capture_jitter_;

  int num_api_calls_in_a_row_ = 0;
  int frames_since_last_report_ = 0;
  bool last_call_was_render_ = false;
  bool proper_call_observed_ = false;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_API_CALL_JITTER_METRICS_H_
