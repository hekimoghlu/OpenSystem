/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#include "sdk/media_constraints.h"

#include "test/gtest.h"

namespace webrtc {

namespace {

// Checks all settings touched by CopyConstraintsIntoRtcConfiguration,
// plus audio_jitter_buffer_max_packets.
bool Matches(const PeerConnectionInterface::RTCConfiguration& a,
             const PeerConnectionInterface::RTCConfiguration& b) {
  return a.audio_jitter_buffer_max_packets ==
             b.audio_jitter_buffer_max_packets &&
         a.screencast_min_bitrate == b.screencast_min_bitrate &&
         a.media_config == b.media_config;
}

TEST(MediaConstraints, CopyConstraintsIntoRtcConfiguration) {
  const MediaConstraints constraints_empty;
  PeerConnectionInterface::RTCConfiguration old_configuration;
  PeerConnectionInterface::RTCConfiguration configuration;

  CopyConstraintsIntoRtcConfiguration(&constraints_empty, &configuration);
  EXPECT_TRUE(Matches(old_configuration, configuration));

  const MediaConstraints constraints_screencast(
      {MediaConstraints::Constraint(MediaConstraints::kScreencastMinBitrate,
                                    "27")},
      {});
  CopyConstraintsIntoRtcConfiguration(&constraints_screencast, &configuration);
  EXPECT_TRUE(configuration.screencast_min_bitrate);
  EXPECT_EQ(27, *(configuration.screencast_min_bitrate));

  // An empty set of constraints will not overwrite
  // values that are already present.
  configuration = old_configuration;
  configuration.audio_jitter_buffer_max_packets = 34;
  CopyConstraintsIntoRtcConfiguration(&constraints_empty, &configuration);
  EXPECT_EQ(34, configuration.audio_jitter_buffer_max_packets);
}

}  // namespace

}  // namespace webrtc
