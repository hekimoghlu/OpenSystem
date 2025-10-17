/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#ifndef API_NETWORK_STATE_PREDICTOR_H_
#define API_NETWORK_STATE_PREDICTOR_H_

#include <cstdint>
#include <memory>

#include "api/transport/bandwidth_usage.h"

namespace webrtc {

// TODO(yinwa): work in progress. API in class NetworkStatePredictor should not
// be used by other users until this comment is removed.

// NetworkStatePredictor predict network state based on current network metrics.
// Usage:
// Setup by calling Initialize.
// For each update, call Update. Update returns network state
// prediction.
class NetworkStatePredictor {
 public:
  virtual ~NetworkStatePredictor() {}

  // Returns current network state prediction.
  // Inputs:  send_time_ms - packet send time.
  //          arrival_time_ms - packet arrival time.
  //          network_state - computed network state.
  virtual BandwidthUsage Update(int64_t send_time_ms,
                                int64_t arrival_time_ms,
                                BandwidthUsage network_state) = 0;
};

class NetworkStatePredictorFactoryInterface {
 public:
  virtual std::unique_ptr<NetworkStatePredictor>
  CreateNetworkStatePredictor() = 0;
  virtual ~NetworkStatePredictorFactoryInterface() = default;
};

}  // namespace webrtc

#endif  // API_NETWORK_STATE_PREDICTOR_H_
