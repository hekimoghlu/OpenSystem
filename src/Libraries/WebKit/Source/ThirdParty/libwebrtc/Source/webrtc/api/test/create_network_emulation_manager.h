/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
#ifndef API_TEST_CREATE_NETWORK_EMULATION_MANAGER_H_
#define API_TEST_CREATE_NETWORK_EMULATION_MANAGER_H_

#include <memory>

#include "api/field_trials_view.h"
#include "api/test/network_emulation_manager.h"

namespace webrtc {

// Returns a non-null NetworkEmulationManager instance.
std::unique_ptr<NetworkEmulationManager> CreateNetworkEmulationManager(
    NetworkEmulationManagerConfig config = NetworkEmulationManagerConfig());

[[deprecated("Use version with NetworkEmulationManagerConfig)")]]
std::unique_ptr<NetworkEmulationManager>
CreateNetworkEmulationManager(
    TimeMode time_mode,
    EmulatedNetworkStatsGatheringMode stats_gathering_mode =
        EmulatedNetworkStatsGatheringMode::kDefault,
    const FieldTrialsView* field_trials = nullptr);

}  // namespace webrtc

#endif  // API_TEST_CREATE_NETWORK_EMULATION_MANAGER_H_
