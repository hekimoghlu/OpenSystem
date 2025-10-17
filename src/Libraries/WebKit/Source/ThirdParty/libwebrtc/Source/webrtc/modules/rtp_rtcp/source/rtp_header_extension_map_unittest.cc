/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"

#include <vector>

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "test/gtest.h"

namespace webrtc {

TEST(RtpHeaderExtensionTest, RegisterByType) {
  RtpHeaderExtensionMap map;
  EXPECT_FALSE(map.IsRegistered(TransmissionOffset::kId));

  EXPECT_TRUE(map.RegisterByType(3, TransmissionOffset::kId));

  EXPECT_TRUE(map.IsRegistered(TransmissionOffset::kId));
  EXPECT_EQ(3, map.GetId(TransmissionOffset::kId));
  EXPECT_EQ(TransmissionOffset::kId, map.GetType(3));
}

TEST(RtpHeaderExtensionTest, RegisterByUri) {
  RtpHeaderExtensionMap map;

  EXPECT_TRUE(map.RegisterByUri(3, TransmissionOffset::Uri()));

  EXPECT_TRUE(map.IsRegistered(TransmissionOffset::kId));
  EXPECT_EQ(3, map.GetId(TransmissionOffset::kId));
  EXPECT_EQ(TransmissionOffset::kId, map.GetType(3));
}

TEST(RtpHeaderExtensionTest, RegisterWithTrait) {
  RtpHeaderExtensionMap map;

  EXPECT_TRUE(map.Register<TransmissionOffset>(3));

  EXPECT_TRUE(map.IsRegistered(TransmissionOffset::kId));
  EXPECT_EQ(3, map.GetId(TransmissionOffset::kId));
  EXPECT_EQ(TransmissionOffset::kId, map.GetType(3));
}

TEST(RtpHeaderExtensionTest, RegisterDuringContruction) {
  const std::vector<RtpExtension> config = {{TransmissionOffset::Uri(), 1},
                                            {AbsoluteSendTime::Uri(), 3}};
  const RtpHeaderExtensionMap map(config);

  EXPECT_EQ(1, map.GetId(TransmissionOffset::kId));
  EXPECT_EQ(3, map.GetId(AbsoluteSendTime::kId));
}

TEST(RtpHeaderExtensionTest, RegisterTwoByteHeaderExtensions) {
  RtpHeaderExtensionMap map;
  // Two-byte header extension needed for id: [15-255].
  EXPECT_TRUE(map.Register<TransmissionOffset>(18));
  EXPECT_TRUE(map.Register<AbsoluteSendTime>(255));
}

TEST(RtpHeaderExtensionTest, RegisterIllegalArg) {
  RtpHeaderExtensionMap map;
  // Valid range for id: [1-255].
  EXPECT_FALSE(map.Register<TransmissionOffset>(0));
  EXPECT_FALSE(map.Register<TransmissionOffset>(256));
}

TEST(RtpHeaderExtensionTest, Idempotent) {
  RtpHeaderExtensionMap map;

  EXPECT_TRUE(map.Register<TransmissionOffset>(3));
  EXPECT_TRUE(map.Register<TransmissionOffset>(3));

  map.Deregister(TransmissionOffset::Uri());
  map.Deregister(TransmissionOffset::Uri());
}

TEST(RtpHeaderExtensionTest, NonUniqueId) {
  RtpHeaderExtensionMap map;
  EXPECT_TRUE(map.Register<TransmissionOffset>(3));

  EXPECT_FALSE(map.Register<AudioLevelExtension>(3));
  EXPECT_TRUE(map.Register<AudioLevelExtension>(4));
}

TEST(RtpHeaderExtensionTest, GetType) {
  RtpHeaderExtensionMap map;
  EXPECT_EQ(RtpHeaderExtensionMap::kInvalidType, map.GetType(3));
  EXPECT_TRUE(map.Register<TransmissionOffset>(3));

  EXPECT_EQ(TransmissionOffset::kId, map.GetType(3));
}

TEST(RtpHeaderExtensionTest, GetId) {
  RtpHeaderExtensionMap map;
  EXPECT_EQ(RtpHeaderExtensionMap::kInvalidId,
            map.GetId(TransmissionOffset::kId));
  EXPECT_TRUE(map.Register<TransmissionOffset>(3));

  EXPECT_EQ(3, map.GetId(TransmissionOffset::kId));
}

TEST(RtpHeaderExtensionTest, RemapFails) {
  RtpHeaderExtensionMap map;
  EXPECT_TRUE(map.Register<TransmissionOffset>(3));
  EXPECT_FALSE(map.Register<TransmissionOffset>(4));
}

}  // namespace webrtc
