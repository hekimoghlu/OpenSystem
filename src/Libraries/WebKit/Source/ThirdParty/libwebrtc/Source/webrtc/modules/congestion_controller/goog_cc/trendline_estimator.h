/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_TRENDLINE_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_TRENDLINE_ESTIMATOR_H_

#include <stddef.h>
#include <stdint.h>

#include <deque>
#include <memory>

#include "api/field_trials_view.h"
#include "api/network_state_predictor.h"
#include "api/transport/bandwidth_usage.h"
#include "modules/congestion_controller/goog_cc/delay_increase_detector_interface.h"
#include "rtc_base/experiments/struct_parameters_parser.h"

namespace webrtc {

struct TrendlineEstimatorSettings {
  static constexpr char kKey[] = "WebRTC-Bwe-TrendlineEstimatorSettings";
  static constexpr unsigned kDefaultTrendlineWindowSize = 20;

  TrendlineEstimatorSettings() = delete;
  explicit TrendlineEstimatorSettings(const FieldTrialsView* key_value_config);

  // Sort the packets in the window. Should be redundant,
  // but then almost no cost.
  bool enable_sort = false;

  // Cap the trendline slope based on the minimum delay seen
  // in the beginning_packets and end_packets respectively.
  bool enable_cap = false;
  unsigned beginning_packets = 7;
  unsigned end_packets = 7;
  double cap_uncertainty = 0.0;

  // Size (in packets) of the window.
  unsigned window_size = kDefaultTrendlineWindowSize;

  std::unique_ptr<StructParametersParser> Parser();
};

class TrendlineEstimator : public DelayIncreaseDetectorInterface {
 public:
  TrendlineEstimator(const FieldTrialsView* key_value_config,
                     NetworkStatePredictor* network_state_predictor);

  ~TrendlineEstimator() override;

  TrendlineEstimator(const TrendlineEstimator&) = delete;
  TrendlineEstimator& operator=(const TrendlineEstimator&) = delete;

  // Update the estimator with a new sample. The deltas should represent deltas
  // between timestamp groups as defined by the InterArrival class.
  void Update(double recv_delta_ms,
              double send_delta_ms,
              int64_t send_time_ms,
              int64_t arrival_time_ms,
              size_t packet_size,
              bool calculated_deltas) override;

  void UpdateTrendline(double recv_delta_ms,
                       double send_delta_ms,
                       int64_t send_time_ms,
                       int64_t arrival_time_ms,
                       size_t packet_size);

  BandwidthUsage State() const override;

  struct PacketTiming {
    PacketTiming(double arrival_time_ms,
                 double smoothed_delay_ms,
                 double raw_delay_ms)
        : arrival_time_ms(arrival_time_ms),
          smoothed_delay_ms(smoothed_delay_ms),
          raw_delay_ms(raw_delay_ms) {}
    double arrival_time_ms;
    double smoothed_delay_ms;
    double raw_delay_ms;
  };

 private:
  friend class GoogCcStatePrinter;
  void Detect(double trend, double ts_delta, int64_t now_ms);

  void UpdateThreshold(double modified_trend, int64_t now_ms);

  // Parameters.
  TrendlineEstimatorSettings settings_;
  const double smoothing_coef_;
  const double threshold_gain_;
  // Used by the existing threshold.
  int num_of_deltas_;
  // Keep the arrival times small by using the change from the first packet.
  int64_t first_arrival_time_ms_;
  // Exponential backoff filtering.
  double accumulated_delay_;
  double smoothed_delay_;
  // Linear least squares regression.
  std::deque<PacketTiming> delay_hist_;

  const double k_up_;
  const double k_down_;
  double overusing_time_threshold_;
  double threshold_;
  double prev_modified_trend_;
  int64_t last_update_ms_;
  double prev_trend_;
  double time_over_using_;
  int overuse_counter_;
  BandwidthUsage hypothesis_;
  BandwidthUsage hypothesis_predicted_;
  NetworkStatePredictor* network_state_predictor_;
};
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_TRENDLINE_ESTIMATOR_H_
