/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MULTI_CHANNEL_CONTENT_DETECTOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MULTI_CHANNEL_CONTENT_DETECTOR_H_

#include <stddef.h>

#include <memory>
#include <optional>
#include <vector>

namespace webrtc {

// Analyzes audio content to determine whether the contained audio is proper
// multichannel, or only upmixed mono. To allow for differences introduced by
// hardware drivers, a threshold `detection_threshold` is used for the
// detection.
// Logs metrics continously and upon destruction.
class MultiChannelContentDetector {
 public:
  // If |stereo_detection_timeout_threshold_seconds| <= 0, no timeout is
  // applied: Once multichannel is detected, the detector remains in that state
  // for its lifetime.
  MultiChannelContentDetector(bool detect_stereo_content,
                              int num_render_input_channels,
                              float detection_threshold,
                              int stereo_detection_timeout_threshold_seconds,
                              float stereo_detection_hysteresis_seconds);

  // Compares the left and right channels in the render `frame` to determine
  // whether the signal is a proper multichannel signal. Returns a bool
  // indicating whether a change in the proper multichannel content was
  // detected.
  bool UpdateDetection(
      const std::vector<std::vector<std::vector<float>>>& frame);

  bool IsProperMultiChannelContentDetected() const {
    return persistent_multichannel_content_detected_;
  }

  bool IsTemporaryMultiChannelContentDetected() const {
    return temporary_multichannel_content_detected_;
  }

 private:
  // Tracks and logs metrics for the amount of multichannel content detected.
  class MetricsLogger {
   public:
    MetricsLogger();

    // The destructor logs call summary statistics.
    ~MetricsLogger();

    // Updates and logs metrics.
    void Update(bool persistent_multichannel_content_detected);

   private:
    int frame_counter_ = 0;

    // Counts the number of frames of persistent multichannel audio observed
    // during the current metrics collection interval.
    int persistent_multichannel_frame_counter_ = 0;

    // Indicates whether persistent multichannel content has ever been detected.
    bool any_multichannel_content_detected_ = false;
  };

  const bool detect_stereo_content_;
  const float detection_threshold_;
  const std::optional<int> detection_timeout_threshold_frames_;
  const int stereo_detection_hysteresis_frames_;

  // Collects and reports metrics on the amount of multichannel content
  // detected. Only created if |num_render_input_channels| > 1 and
  // |detect_stereo_content_| is true.
  const std::unique_ptr<MetricsLogger> metrics_logger_;

  bool persistent_multichannel_content_detected_;
  bool temporary_multichannel_content_detected_ = false;
  int64_t frames_since_stereo_detected_last_ = 0;
  int64_t consecutive_frames_with_stereo_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MULTI_CHANNEL_CONTENT_DETECTOR_H_
