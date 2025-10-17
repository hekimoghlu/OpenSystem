/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include "net/dcsctp/packet/parameter/zero_checksum_acceptable_chunk_parameter.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(ZeroChecksumAcceptableChunkParameterTest, SerializeAndDeserialize) {
  ZeroChecksumAcceptableChunkParameter parameter(
      ZeroChecksumAlternateErrorDetectionMethod::LowerLayerDtls());

  std::vector<uint8_t> serialized;
  parameter.SerializeTo(serialized);

  EXPECT_THAT(serialized,
              ElementsAre(0x80, 0x01, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01));

  ASSERT_HAS_VALUE_AND_ASSIGN(
      ZeroChecksumAcceptableChunkParameter deserialized,
      ZeroChecksumAcceptableChunkParameter::Parse(serialized));
}

TEST(ZeroChecksumAcceptableChunkParameterTest, FailToDeserializePrevVersion) {
  // This is how the draft described the chunk as, in version 00.
  std::vector<uint8_t> invalid = {0x80, 0x01, 0x00, 0x04};

  EXPECT_FALSE(
      ZeroChecksumAcceptableChunkParameter::Parse(invalid).has_value());
}

TEST(ZeroChecksumAcceptableChunkParameterTest, FailToDeserialize) {
  std::vector<uint8_t> invalid = {0x00, 0x00, 0x00, 0x00};

  EXPECT_FALSE(
      ZeroChecksumAcceptableChunkParameter::Parse(invalid).has_value());
}

TEST(ZeroChecksumAcceptableChunkParameterTest, HasToString) {
  ZeroChecksumAcceptableChunkParameter parameter(
      ZeroChecksumAlternateErrorDetectionMethod::LowerLayerDtls());

  EXPECT_EQ(parameter.ToString(), "Zero Checksum Acceptable (1)");
}

}  // namespace
}  // namespace dcsctp
