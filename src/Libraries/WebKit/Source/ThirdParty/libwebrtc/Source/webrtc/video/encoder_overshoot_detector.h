/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#ifndef VIDEO_ENCODER_OVERSHOOT_DETECTOR_H_
#define VIDEO_ENCODER_OVERSHOOT_DETECTOR_H_

#include <deque>
#include <optional>

#include "api/units/data_rate.h"
#include "api/video_codecs/video_codec.h"

namespace webrtc {

class EncoderOvershootDetector {
 public:
  explicit EncoderOvershootDetector(int64_t window_size_ms,
                                    VideoCodecType codec,
                                    bool is_screenshare);
  ~EncoderOvershootDetector();

  void SetTargetRate(DataRate target_bitrate,
                     double target_framerate_fps,
                     int64_t time_ms);
  // A frame has been encoded or dropped. `bytes` == 0 indicates a drop.
  void OnEncodedFrame(size_t bytes, int64_t time_ms);
  // This utilization factor reaches 1.0 only if the encoder produces encoded
  // frame in such a way that they can be sent onto the network at
  // `target_bitrate` without building growing queues.
  std::optional<double> GetNetworkRateUtilizationFactor(int64_t time_ms);
  // This utilization factor is based just on actual encoded frame sizes in
  // relation to ideal sizes. An undershoot may be compensated by an
  // overshoot so that the average over time is close to `target_bitrate`.
  std::optional<double> GetMediaRateUtilizationFactor(int64_t time_ms);
  void Reset();

 private:
  int64_t IdealFrameSizeBits() const;
  void LeakBits(int64_t time_ms);
  void CullOldUpdates(int64_t time_ms);
  // Updates provided buffer and checks if overuse ensues, returns
  // the calculated utilization factor for this frame.
  double HandleEncodedFrame(size_t frame_size_bits,
                            int64_t ideal_frame_size_bits,
                            int64_t time_ms,
                            int64_t* buffer_level_bits) const;

  const int64_t window_size_ms_;
  int64_t time_last_update_ms_;
  struct BitrateUpdate {
    BitrateUpdate(double network_utilization_factor,
                  double media_utilization_factor,
                  int64_t update_time_ms)
        : network_utilization_factor(network_utilization_factor),
          media_utilization_factor(media_utilization_factor),
          update_time_ms(update_time_ms) {}
    // The utilization factor based on strict network rate.
    double network_utilization_factor;
    // The utilization based on average media rate.
    double media_utilization_factor;
    int64_t update_time_ms;
  };
  void UpdateHistograms();
  std::deque<BitrateUpdate> utilization_factors_;
  double sum_network_utilization_factors_;
  double sum_media_utilization_factors_;
  DataRate target_bitrate_;
  double target_framerate_fps_;
  int64_t network_buffer_level_bits_;
  int64_t media_buffer_level_bits_;
  VideoCodecType codec_;
  bool is_screenshare_;
  int64_t frame_count_;
  int64_t sum_diff_kbps_squared_;
  int64_t sum_overshoot_percent_;
};

}  // namespace webrtc

#endif  // VIDEO_ENCODER_OVERSHOOT_DETECTOR_H_
