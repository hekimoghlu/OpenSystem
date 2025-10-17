/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#ifndef P2P_BASE_ACTIVE_ICE_CONTROLLER_FACTORY_INTERFACE_H_
#define P2P_BASE_ACTIVE_ICE_CONTROLLER_FACTORY_INTERFACE_H_

#include <memory>

#include "p2p/base/active_ice_controller_interface.h"
#include "p2p/base/ice_agent_interface.h"
#include "p2p/base/ice_controller_factory_interface.h"

namespace cricket {

// An active ICE controller may be constructed with the same arguments as a
// legacy ICE controller. Additionally, an ICE agent must be provided for the
// active ICE controller to interact with.
struct ActiveIceControllerFactoryArgs {
  IceControllerFactoryArgs legacy_args;
  IceAgentInterface* ice_agent;
};

class ActiveIceControllerFactoryInterface {
 public:
  virtual ~ActiveIceControllerFactoryInterface() = default;
  virtual std::unique_ptr<ActiveIceControllerInterface> Create(
      const ActiveIceControllerFactoryArgs&) = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_ACTIVE_ICE_CONTROLLER_FACTORY_INTERFACE_H_
