/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include <array>
#include <vector>

#include "api/audio/audio_processing.h"
#include "modules/audio_processing/echo_control_mobile_impl.h"
#include "test/gtest.h"

namespace webrtc {
TEST(EchoControlMobileTest, InterfaceConfiguration) {
  EchoControlMobileImpl aecm;
  aecm.Initialize(AudioProcessing::kSampleRate16kHz, 2, 2);

  // Toggle routing modes
  std::array<EchoControlMobileImpl::RoutingMode, 5> routing_modes = {
      EchoControlMobileImpl::kQuietEarpieceOrHeadset,
      EchoControlMobileImpl::kEarpiece,
      EchoControlMobileImpl::kLoudEarpiece,
      EchoControlMobileImpl::kSpeakerphone,
      EchoControlMobileImpl::kLoudSpeakerphone,
  };
  for (auto mode : routing_modes) {
    EXPECT_EQ(0, aecm.set_routing_mode(mode));
    EXPECT_EQ(mode, aecm.routing_mode());
  }

  // Turn comfort noise off/on
  EXPECT_EQ(0, aecm.enable_comfort_noise(false));
  EXPECT_FALSE(aecm.is_comfort_noise_enabled());
  EXPECT_EQ(0, aecm.enable_comfort_noise(true));
  EXPECT_TRUE(aecm.is_comfort_noise_enabled());
}

}  // namespace webrtc
