/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#include "api/transport/goog_cc_factory.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "api/transport/network_control.h"
#include "api/units/time_delta.h"
#include "modules/congestion_controller/goog_cc/goog_cc_network_control.h"

namespace webrtc {

GoogCcNetworkControllerFactory::GoogCcNetworkControllerFactory(
    GoogCcFactoryConfig config)
    : factory_config_(std::move(config)) {}

std::unique_ptr<NetworkControllerInterface>
GoogCcNetworkControllerFactory::Create(NetworkControllerConfig config) {
  GoogCcConfig goog_cc_config;
  goog_cc_config.feedback_only = factory_config_.feedback_only;
  if (factory_config_.network_state_estimator_factory) {
    goog_cc_config.network_state_estimator =
        factory_config_.network_state_estimator_factory->Create(
            &config.env.field_trials());
  }
  if (factory_config_.network_state_predictor_factory) {
    goog_cc_config.network_state_predictor =
        factory_config_.network_state_predictor_factory
            ->CreateNetworkStatePredictor();
  }
  return std::make_unique<GoogCcNetworkController>(config,
                                                   std::move(goog_cc_config));
}

TimeDelta GoogCcNetworkControllerFactory::GetProcessInterval() const {
  const int64_t kUpdateIntervalMs = 25;
  return TimeDelta::Millis(kUpdateIntervalMs);
}

}  // namespace webrtc
