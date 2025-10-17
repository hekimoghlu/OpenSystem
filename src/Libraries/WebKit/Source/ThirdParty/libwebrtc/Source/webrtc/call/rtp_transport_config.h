/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#ifndef CALL_RTP_TRANSPORT_CONFIG_H_
#define CALL_RTP_TRANSPORT_CONFIG_H_

#include <optional>

#include "api/environment/environment.h"
#include "api/network_state_predictor.h"
#include "api/transport/bitrate_settings.h"
#include "api/transport/network_control.h"
#include "api/units/time_delta.h"

namespace webrtc {

struct RtpTransportConfig {
  Environment env;

  // Bitrate config used until valid bitrate estimates are calculated. Also
  // used to cap total bitrate used. This comes from the remote connection.
  BitrateConstraints bitrate_config;

  // NetworkStatePredictor to use for this call.
  NetworkStatePredictorFactoryInterface* network_state_predictor_factory =
      nullptr;

  // Network controller factory to use for this call.
  NetworkControllerFactoryInterface* network_controller_factory = nullptr;

  // The burst interval of the pacer, see TaskQueuePacedSender constructor.
  std::optional<TimeDelta> pacer_burst_interval;
};
}  // namespace webrtc

#endif  // CALL_RTP_TRANSPORT_CONFIG_H_
