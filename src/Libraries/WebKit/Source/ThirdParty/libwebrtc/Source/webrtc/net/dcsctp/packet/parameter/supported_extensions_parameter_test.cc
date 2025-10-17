/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#include "net/dcsctp/packet/parameter/supported_extensions_parameter.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(SupportedExtensionsParameterTest, SerializeAndDeserialize) {
  SupportedExtensionsParameter parameter({1, 2, 3});

  std::vector<uint8_t> serialized;
  parameter.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(SupportedExtensionsParameter deserialized,
                              SupportedExtensionsParameter::Parse(serialized));

  EXPECT_THAT(deserialized.chunk_types(), ElementsAre(1, 2, 3));
  EXPECT_TRUE(deserialized.supports(1));
  EXPECT_TRUE(deserialized.supports(2));
  EXPECT_TRUE(deserialized.supports(3));
  EXPECT_FALSE(deserialized.supports(4));
}

}  // namespace
}  // namespace dcsctp
