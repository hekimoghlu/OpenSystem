/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#ifndef RTC_BASE_EXPERIMENTS_RATE_CONTROL_SETTINGS_H_
#define RTC_BASE_EXPERIMENTS_RATE_CONTROL_SETTINGS_H_

#include <optional>

#include "api/field_trials_view.h"
#include "api/units/data_size.h"
#include "api/video_codecs/video_codec.h"
#include "rtc_base/experiments/struct_parameters_parser.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {

struct CongestionWindowConfig {
  static constexpr char kKey[] = "WebRTC-CongestionWindow";
  std::optional<int> queue_size_ms;
  std::optional<int> min_bitrate_bps;
  std::optional<DataSize> initial_data_window;
  bool drop_frame_only = false;
  std::unique_ptr<StructParametersParser> Parser();
  static CongestionWindowConfig Parse(absl::string_view config);
};

struct VideoRateControlConfig {
  static constexpr char kKey[] = "WebRTC-VideoRateControl";
  std::optional<double> pacing_factor;
  bool alr_probing = false;
  std::optional<int> vp8_qp_max;
  std::optional<int> vp8_min_pixels;
  bool trust_vp8 = true;
  bool trust_vp9 = true;
  bool bitrate_adjuster = true;
  bool adjuster_use_headroom = true;
  bool vp8_s0_boost = false;
  bool vp8_base_heavy_tl3_alloc = false;

  std::unique_ptr<StructParametersParser> Parser();
};

class RateControlSettings final {
 public:
  explicit RateControlSettings(const FieldTrialsView& key_value_config);
  RateControlSettings(RateControlSettings&&);
  ~RateControlSettings();

  // When CongestionWindowPushback is enabled, the pacer is oblivious to
  // the congestion window. The relation between outstanding data and
  // the congestion window affects encoder allocations directly.
  bool UseCongestionWindow() const;
  int64_t GetCongestionWindowAdditionalTimeMs() const;
  bool UseCongestionWindowPushback() const;
  bool UseCongestionWindowDropFrameOnly() const;
  uint32_t CongestionWindowMinPushbackTargetBitrateBps() const;
  std::optional<DataSize> CongestionWindowInitialDataWindow() const;

  std::optional<double> GetPacingFactor() const;
  bool UseAlrProbing() const;

  std::optional<int> LibvpxVp8QpMax() const;
  std::optional<int> LibvpxVp8MinPixels() const;
  bool LibvpxVp8TrustedRateController() const;
  bool Vp8BoostBaseLayerQuality() const;
  bool Vp8DynamicRateSettings() const;
  bool LibvpxVp9TrustedRateController() const;
  bool Vp9DynamicRateSettings() const;

  bool Vp8BaseHeavyTl3RateAllocation() const;

  bool UseEncoderBitrateAdjuster() const;
  bool BitrateAdjusterCanUseNetworkHeadroom() const;

 private:
  CongestionWindowConfig congestion_window_config_;
  VideoRateControlConfig video_config_;
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_RATE_CONTROL_SETTINGS_H_
