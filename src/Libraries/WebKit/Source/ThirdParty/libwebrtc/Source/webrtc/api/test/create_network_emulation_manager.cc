/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#include "api/test/create_network_emulation_manager.h"

#include <memory>
#include <utility>

#include "api/field_trials_view.h"
#include "api/test/network_emulation_manager.h"
#include "test/network/network_emulation_manager.h"

namespace webrtc {

std::unique_ptr<NetworkEmulationManager> CreateNetworkEmulationManager(
    NetworkEmulationManagerConfig config) {
  return std::make_unique<test::NetworkEmulationManagerImpl>(std::move(config));
}

std::unique_ptr<NetworkEmulationManager> CreateNetworkEmulationManager(
    TimeMode time_mode,
    EmulatedNetworkStatsGatheringMode stats_gathering_mode,
    const FieldTrialsView* field_trials) {
  return CreateNetworkEmulationManager(
      {.time_mode = time_mode,
       .stats_gathering_mode = stats_gathering_mode,
       .field_trials = field_trials});
}

}  // namespace webrtc
