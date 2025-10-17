/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#include "call/call_config.h"

#include "api/environment/environment.h"
#include "api/task_queue/task_queue_base.h"
#include "call/rtp_transport_config.h"

namespace webrtc {

CallConfig::CallConfig(const Environment& env,
                       TaskQueueBase* network_task_queue)
    : env(env),
      network_task_queue_(network_task_queue) {}

RtpTransportConfig CallConfig::ExtractTransportConfig() const {
  RtpTransportConfig transport_config = {.env = env};
  transport_config.bitrate_config = bitrate_config;
  transport_config.network_controller_factory =
      per_call_network_controller_factory
          ? per_call_network_controller_factory.get()
          : network_controller_factory;
  transport_config.network_state_predictor_factory =
      network_state_predictor_factory;
  transport_config.pacer_burst_interval = pacer_burst_interval;

  return transport_config;
}

CallConfig::~CallConfig() = default;

}  // namespace webrtc
