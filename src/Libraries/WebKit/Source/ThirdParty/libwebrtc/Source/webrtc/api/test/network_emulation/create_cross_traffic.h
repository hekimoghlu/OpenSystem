/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#ifndef API_TEST_NETWORK_EMULATION_CREATE_CROSS_TRAFFIC_H_
#define API_TEST_NETWORK_EMULATION_CREATE_CROSS_TRAFFIC_H_

#include <memory>

#include "api/test/network_emulation/cross_traffic.h"
#include "api/test/network_emulation_manager.h"

namespace webrtc {

// This API is still in development and can be changed without prior notice.

std::unique_ptr<CrossTrafficGenerator> CreateRandomWalkCrossTraffic(
    CrossTrafficRoute* traffic_route,
    RandomWalkConfig config);

std::unique_ptr<CrossTrafficGenerator> CreatePulsedPeaksCrossTraffic(
    CrossTrafficRoute* traffic_route,
    PulsedPeaksConfig config);

std::unique_ptr<CrossTrafficGenerator> CreateFakeTcpCrossTraffic(
    EmulatedRoute* send_route,
    EmulatedRoute* ret_route,
    FakeTcpConfig config);

}  // namespace webrtc

#endif  // API_TEST_NETWORK_EMULATION_CREATE_CROSS_TRAFFIC_H_
