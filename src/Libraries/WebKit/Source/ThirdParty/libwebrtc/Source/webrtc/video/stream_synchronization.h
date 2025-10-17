/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#ifndef VIDEO_STREAM_SYNCHRONIZATION_H_
#define VIDEO_STREAM_SYNCHRONIZATION_H_

#include <stdint.h>

#include "system_wrappers/include/rtp_to_ntp_estimator.h"

namespace webrtc {

class StreamSynchronization {
 public:
  struct Measurements {
    Measurements() : latest_receive_time_ms(0), latest_timestamp(0) {}
    RtpToNtpEstimator rtp_to_ntp;
    int64_t latest_receive_time_ms;
    uint32_t latest_timestamp;
  };

  StreamSynchronization(uint32_t video_stream_id, uint32_t audio_stream_id);

  bool ComputeDelays(int relative_delay_ms,
                     int current_audio_delay_ms,
                     int* total_audio_delay_target_ms,
                     int* total_video_delay_target_ms);

  // On success `relative_delay_ms` contains the number of milliseconds later
  // video is rendered relative audio. If audio is played back later than video
  // `relative_delay_ms` will be negative.
  static bool ComputeRelativeDelay(const Measurements& audio_measurement,
                                   const Measurements& video_measurement,
                                   int* relative_delay_ms);

  // Set target buffering delay. Audio and video will be delayed by at least
  // `target_delay_ms`.
  void SetTargetBufferingDelay(int target_delay_ms);

  // Lowers the audio delay by 10%. Can be used to recover from errors.
  void ReduceAudioDelay();

  // Lowers the video delay by 10%. Can be used to recover from errors.
  void ReduceVideoDelay();

  uint32_t audio_stream_id() const { return audio_stream_id_; }
  uint32_t video_stream_id() const { return video_stream_id_; }

 private:
  struct SynchronizationDelays {
    int extra_ms = 0;
    int last_ms = 0;
  };

  const uint32_t video_stream_id_;
  const uint32_t audio_stream_id_;
  SynchronizationDelays audio_delay_;
  SynchronizationDelays video_delay_;
  int base_target_delay_ms_;
  int avg_diff_ms_;
};
}  // namespace webrtc

#endif  // VIDEO_STREAM_SYNCHRONIZATION_H_
