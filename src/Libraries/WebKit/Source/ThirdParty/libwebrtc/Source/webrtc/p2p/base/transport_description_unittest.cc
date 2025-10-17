/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "p2p/base/transport_description.h"

#include "test/gtest.h"

using webrtc::RTCErrorType;

namespace cricket {

TEST(IceParameters, SuccessfulParse) {
  auto result = IceParameters::Parse("ufrag", "22+characters+long+pwd");
  ASSERT_TRUE(result.ok());
  IceParameters parameters = result.MoveValue();
  EXPECT_EQ("ufrag", parameters.ufrag);
  EXPECT_EQ("22+characters+long+pwd", parameters.pwd);
}

TEST(IceParameters, FailedParseShortUfrag) {
  auto result = IceParameters::Parse("3ch", "22+characters+long+pwd");
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

TEST(IceParameters, FailedParseLongUfrag) {
  std::string ufrag(257, '+');
  auto result = IceParameters::Parse(ufrag, "22+characters+long+pwd");
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

TEST(IceParameters, FailedParseShortPwd) {
  auto result = IceParameters::Parse("ufrag", "21+character+long+pwd");
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

TEST(IceParameters, FailedParseLongPwd) {
  std::string pwd(257, '+');
  auto result = IceParameters::Parse("ufrag", pwd);
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

TEST(IceParameters, FailedParseBadUfragChar) {
  auto result = IceParameters::Parse("ufrag\r\n", "22+characters+long+pwd");
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

TEST(IceParameters, FailedParseBadPwdChar) {
  auto result = IceParameters::Parse("ufrag", "22+characters+long+pwd\r\n");
  EXPECT_EQ(RTCErrorType::SYNTAX_ERROR, result.error().type());
}

}  // namespace cricket
