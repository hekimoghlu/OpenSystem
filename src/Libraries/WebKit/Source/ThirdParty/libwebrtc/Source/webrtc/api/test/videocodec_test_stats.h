/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#ifndef API_TEST_VIDEOCODEC_TEST_STATS_H_
#define API_TEST_VIDEOCODEC_TEST_STATS_H_

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <string>
#include <vector>

#include "api/units/data_rate.h"
#include "api/units/frequency.h"
#include "api/video/video_frame_type.h"

namespace webrtc {
namespace test {

// Statistics for a sequence of processed frames. This class is not thread safe.
// TODO(webrtc:14852): Deprecated in favor VideoCodecStats.
class VideoCodecTestStats {
 public:
  // Statistics for one processed frame.
  struct FrameStatistics {
    FrameStatistics(size_t frame_number,
                    size_t rtp_timestamp,
                    size_t spatial_idx);

    std::string ToString() const;

    // Returns name -> value text map of frame statistics.
    std::map<std::string, std::string> ToMap() const;

    size_t frame_number = 0;
    size_t rtp_timestamp = 0;

    // Encoding.
    int64_t encode_start_ns = 0;
    int encode_return_code = 0;
    bool encoding_successful = false;
    size_t encode_time_us = 0;
    size_t target_bitrate_kbps = 0;
    double target_framerate_fps = 0.0;
    size_t length_bytes = 0;
    VideoFrameType frame_type = VideoFrameType::kVideoFrameDelta;

    // Layering.
    size_t spatial_idx = 0;
    size_t temporal_idx = 0;
    bool inter_layer_predicted = false;
    bool non_ref_for_inter_layer_pred = true;

    // H264 specific.
    size_t max_nalu_size_bytes = 0;

    // Decoding.
    int64_t decode_start_ns = 0;
    int decode_return_code = 0;
    bool decoding_successful = false;
    size_t decode_time_us = 0;
    size_t decoded_width = 0;
    size_t decoded_height = 0;

    // Quantization.
    int qp = -1;

    // Quality.
    bool quality_analysis_successful = false;
    float psnr_y = 0.0f;
    float psnr_u = 0.0f;
    float psnr_v = 0.0f;
    float psnr = 0.0f;  // 10 * log10(255^2 / (mse_y + mse_u + mse_v)).
    float ssim = 0.0f;  // 0.8 * ssim_y + 0.1 * (ssim_u + ssim_v).
  };

  struct VideoStatistics {
    std::string ToString(std::string prefix) const;

    // Returns name -> value text map of video statistics.
    std::map<std::string, std::string> ToMap() const;

    size_t target_bitrate_kbps = 0;
    float input_framerate_fps = 0.0f;

    size_t spatial_idx = 0;
    size_t temporal_idx = 0;

    size_t width = 0;
    size_t height = 0;

    size_t length_bytes = 0;
    size_t bitrate_kbps = 0;
    float framerate_fps = 0;

    float enc_speed_fps = 0.0f;
    float dec_speed_fps = 0.0f;

    float avg_encode_latency_sec = 0.0f;
    float max_encode_latency_sec = 0.0f;
    float avg_decode_latency_sec = 0.0f;
    float max_decode_latency_sec = 0.0f;

    float avg_delay_sec = 0.0f;
    float max_key_frame_delay_sec = 0.0f;
    float max_delta_frame_delay_sec = 0.0f;
    float time_to_reach_target_bitrate_sec = 0.0f;
    float avg_bitrate_mismatch_pct = 0.0f;
    float avg_framerate_mismatch_pct = 0.0f;

    float avg_key_frame_size_bytes = 0.0f;
    float avg_delta_frame_size_bytes = 0.0f;
    float avg_qp = 0.0f;

    float avg_psnr_y = 0.0f;
    float avg_psnr_u = 0.0f;
    float avg_psnr_v = 0.0f;
    float avg_psnr = 0.0f;
    float min_psnr = 0.0f;
    float avg_ssim = 0.0f;
    float min_ssim = 0.0f;

    size_t num_input_frames = 0;
    size_t num_encoded_frames = 0;
    size_t num_decoded_frames = 0;
    size_t num_key_frames = 0;
    size_t num_spatial_resizes = 0;
    size_t max_nalu_size_bytes = 0;
  };

  virtual ~VideoCodecTestStats() = default;

  virtual std::vector<FrameStatistics> GetFrameStatistics() const = 0;

  virtual std::vector<VideoStatistics> SliceAndCalcLayerVideoStatistic(
      size_t first_frame_num,
      size_t last_frame_num) = 0;

  virtual VideoStatistics CalcVideoStatistic(size_t first_frame,
                                             size_t last_frame,
                                             DataRate target_bitrate,
                                             Frequency target_framerate) = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_VIDEOCODEC_TEST_STATS_H_
