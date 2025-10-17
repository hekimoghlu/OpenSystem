/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#ifndef P2P_BASE_ICE_CONTROLLER_FACTORY_INTERFACE_H_
#define P2P_BASE_ICE_CONTROLLER_FACTORY_INTERFACE_H_

#include <memory>
#include <string>

#include "p2p/base/ice_controller_interface.h"
#include "p2p/base/ice_transport_internal.h"

namespace cricket {

// struct with arguments to IceControllerFactoryInterface::Create
struct IceControllerFactoryArgs {
  std::function<IceTransportState()> ice_transport_state_func;
  std::function<IceRole()> ice_role_func;
  std::function<bool(const Connection*)> is_connection_pruned_func;
  const IceFieldTrials* ice_field_trials;
  std::string ice_controller_field_trials;
};

class IceControllerFactoryInterface {
 public:
  virtual ~IceControllerFactoryInterface() = default;
  virtual std::unique_ptr<IceControllerInterface> Create(
      const IceControllerFactoryArgs& args) = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_ICE_CONTROLLER_FACTORY_INTERFACE_H_
