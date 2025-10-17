/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_PROBE_BITRATE_ESTIMATOR_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_PROBE_BITRATE_ESTIMATOR_H_

#include <map>
#include <optional>

#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"

namespace webrtc {
class RtcEventLog;

class ProbeBitrateEstimator {
 public:
  explicit ProbeBitrateEstimator(RtcEventLog* event_log);
  ~ProbeBitrateEstimator();

  // Should be called for every probe packet we receive feedback about.
  // Returns the estimated bitrate if the probe completes a valid cluster.
  std::optional<DataRate> HandleProbeAndEstimateBitrate(
      const PacketResult& packet_feedback);

  std::optional<DataRate> FetchAndResetLastEstimatedBitrate();

 private:
  struct AggregatedCluster {
    int num_probes = 0;
    Timestamp first_send = Timestamp::PlusInfinity();
    Timestamp last_send = Timestamp::MinusInfinity();
    Timestamp first_receive = Timestamp::PlusInfinity();
    Timestamp last_receive = Timestamp::MinusInfinity();
    DataSize size_last_send = DataSize::Zero();
    DataSize size_first_receive = DataSize::Zero();
    DataSize size_total = DataSize::Zero();
  };

  // Erases old cluster data that was seen before `timestamp`.
  void EraseOldClusters(Timestamp timestamp);

  std::map<int, AggregatedCluster> clusters_;
  RtcEventLog* const event_log_;
  std::optional<DataRate> estimated_data_rate_;
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_PROBE_BITRATE_ESTIMATOR_H_
