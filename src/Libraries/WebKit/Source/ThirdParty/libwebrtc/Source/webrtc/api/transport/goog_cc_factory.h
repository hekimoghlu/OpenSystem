/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef API_TRANSPORT_GOOG_CC_FACTORY_H_
#define API_TRANSPORT_GOOG_CC_FACTORY_H_

#include <memory>

#include "api/network_state_predictor.h"
#include "api/transport/network_control.h"
#include "api/units/time_delta.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

struct GoogCcFactoryConfig {
  std::unique_ptr<NetworkStateEstimatorFactory> network_state_estimator_factory;
  NetworkStatePredictorFactoryInterface* network_state_predictor_factory =
      nullptr;
  bool feedback_only = false;
};

class RTC_EXPORT GoogCcNetworkControllerFactory
    : public NetworkControllerFactoryInterface {
 public:
  GoogCcNetworkControllerFactory() = default;
  explicit GoogCcNetworkControllerFactory(GoogCcFactoryConfig config);

  std::unique_ptr<NetworkControllerInterface> Create(
      NetworkControllerConfig config) override;
  TimeDelta GetProcessInterval() const override;

 private:
  GoogCcFactoryConfig factory_config_;
};

}  // namespace webrtc

#endif  // API_TRANSPORT_GOOG_CC_FACTORY_H_
