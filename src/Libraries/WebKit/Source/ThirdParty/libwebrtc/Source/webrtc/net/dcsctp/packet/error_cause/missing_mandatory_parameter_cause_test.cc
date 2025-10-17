/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
#include "net/dcsctp/packet/error_cause/missing_mandatory_parameter_cause.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(MissingMandatoryParameterCauseTest, SerializeAndDeserialize) {
  uint16_t parameter_types[] = {1, 2, 3};
  MissingMandatoryParameterCause parameter(parameter_types);

  std::vector<uint8_t> serialized;
  parameter.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      MissingMandatoryParameterCause deserialized,
      MissingMandatoryParameterCause::Parse(serialized));

  EXPECT_THAT(deserialized.missing_parameter_types(), ElementsAre(1, 2, 3));
}

TEST(MissingMandatoryParameterCauseTest, HandlesDeserializeZeroParameters) {
  uint8_t serialized[] = {0, 2, 0, 8, 0, 0, 0, 0};

  ASSERT_HAS_VALUE_AND_ASSIGN(
      MissingMandatoryParameterCause deserialized,
      MissingMandatoryParameterCause::Parse(serialized));

  EXPECT_THAT(deserialized.missing_parameter_types(), IsEmpty());
}

TEST(MissingMandatoryParameterCauseTest, HandlesOverflowParameterCount) {
  // 0x80000004 * 2 = 2**32 + 8 -> if overflow, would validate correctly.
  uint8_t serialized[] = {0, 2, 0, 8, 0x80, 0x00, 0x00, 0x04};

  EXPECT_FALSE(MissingMandatoryParameterCause::Parse(serialized).has_value());
}

}  // namespace
}  // namespace dcsctp
