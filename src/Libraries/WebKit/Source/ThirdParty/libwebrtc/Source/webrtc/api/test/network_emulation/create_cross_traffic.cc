/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include "api/test/network_emulation/create_cross_traffic.h"

#include <memory>

#include "api/test/network_emulation/cross_traffic.h"
#include "test/network/cross_traffic.h"
#include "test/network/network_emulation.h"

namespace webrtc {

std::unique_ptr<CrossTrafficGenerator> CreateRandomWalkCrossTraffic(
    CrossTrafficRoute* traffic_route,
    RandomWalkConfig config) {
  return std::make_unique<test::RandomWalkCrossTraffic>(config, traffic_route);
}

std::unique_ptr<CrossTrafficGenerator> CreatePulsedPeaksCrossTraffic(
    CrossTrafficRoute* traffic_route,
    PulsedPeaksConfig config) {
  return std::make_unique<test::PulsedPeaksCrossTraffic>(config, traffic_route);
}

std::unique_ptr<CrossTrafficGenerator> CreateFakeTcpCrossTraffic(
    EmulatedRoute* send_route,
    EmulatedRoute* ret_route,
    FakeTcpConfig config) {
  return std::make_unique<test::FakeTcpCrossTraffic>(config, send_route,
                                                     ret_route);
}

}  // namespace webrtc
