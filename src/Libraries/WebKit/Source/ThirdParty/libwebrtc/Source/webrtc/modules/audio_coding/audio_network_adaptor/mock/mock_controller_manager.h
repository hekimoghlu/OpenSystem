/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_CONTROLLER_MANAGER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_CONTROLLER_MANAGER_H_

#include <vector>

#include "modules/audio_coding/audio_network_adaptor/controller_manager.h"
#include "test/gmock.h"

namespace webrtc {

class MockControllerManager : public ControllerManager {
 public:
  ~MockControllerManager() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(std::vector<Controller*>,
              GetSortedControllers,
              (const Controller::NetworkMetrics& metrics),
              (override));
  MOCK_METHOD(std::vector<Controller*>, GetControllers, (), (const, override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_MOCK_MOCK_CONTROLLER_MANAGER_H_
