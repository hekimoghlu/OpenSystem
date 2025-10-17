/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#ifndef MODULES_CONGESTION_CONTROLLER_PCC_PCC_FACTORY_H_
#define MODULES_CONGESTION_CONTROLLER_PCC_PCC_FACTORY_H_

#include <memory>

#include "api/transport/network_control.h"
#include "api/units/time_delta.h"

namespace webrtc {

class PccNetworkControllerFactory : public NetworkControllerFactoryInterface {
 public:
  PccNetworkControllerFactory();
  std::unique_ptr<NetworkControllerInterface> Create(
      NetworkControllerConfig config) override;
  TimeDelta GetProcessInterval() const override;
};
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_PCC_PCC_FACTORY_H_
