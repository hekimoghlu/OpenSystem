/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "modules/rtp_rtcp/source/rtp_header_extension_size.h"

#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "test/gtest.h"

namespace {

using ::webrtc::RtpExtensionSize;
using ::webrtc::RtpHeaderExtensionMap;
using ::webrtc::RtpHeaderExtensionSize;
using ::webrtc::RtpMid;
using ::webrtc::RtpStreamId;

// id for 1-byte header extension. actual value is irrelevant for these tests.
constexpr int kId = 1;
// id that forces to use 2-byte header extension.
constexpr int kIdForceTwoByteHeader = 15;

TEST(RtpHeaderExtensionSizeTest, ReturnsZeroIfNoExtensionsAreRegistered) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 3}};
  // Register different extension than ask size for.
  RtpHeaderExtensionMap registered;
  registered.Register<RtpStreamId>(kId);

  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 0);
}

TEST(RtpHeaderExtensionSizeTest, IncludesSizeOfExtensionHeaders) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 3}};
  RtpHeaderExtensionMap registered;
  registered.Register<RtpMid>(kId);

  // 4 bytes for extension block header + 1 byte for individual extension header
  // + 3 bytes for the value.
  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 8);
}

TEST(RtpHeaderExtensionSizeTest, RoundsUpTo32bitAlignmant) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 5}};
  RtpHeaderExtensionMap registered;
  registered.Register<RtpMid>(kId);

  // 10 bytes of data including headers + 2 bytes of padding.
  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 12);
}

TEST(RtpHeaderExtensionSizeTest, SumsSeveralExtensions) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 16},
                                                  {RtpStreamId::kId, 2}};
  RtpHeaderExtensionMap registered;
  registered.Register<RtpMid>(kId);
  registered.Register<RtpStreamId>(14);

  // 4 bytes for extension block header + 18 bytes of value +
  // 2 bytes for two headers
  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 24);
}

TEST(RtpHeaderExtensionSizeTest, LargeIdForce2BytesHeader) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 3},
                                                  {RtpStreamId::kId, 2}};
  RtpHeaderExtensionMap registered;
  registered.Register<RtpMid>(kId);
  registered.Register<RtpStreamId>(kIdForceTwoByteHeader);

  // 4 bytes for extension block header + 5 bytes of value +
  // 2*2 bytes for two headers + 3 bytes of padding.
  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 16);
}

TEST(RtpHeaderExtensionSizeTest, LargeValueForce2BytesHeader) {
  constexpr RtpExtensionSize kExtensionSizes[] = {{RtpMid::kId, 17},
                                                  {RtpStreamId::kId, 4}};
  RtpHeaderExtensionMap registered;
  registered.Register<RtpMid>(1);
  registered.Register<RtpStreamId>(2);

  // 4 bytes for extension block header + 21 bytes of value +
  // 2*2 bytes for two headers + 3 byte of padding.
  EXPECT_EQ(RtpHeaderExtensionSize(kExtensionSizes, registered), 32);
}

}  // namespace
