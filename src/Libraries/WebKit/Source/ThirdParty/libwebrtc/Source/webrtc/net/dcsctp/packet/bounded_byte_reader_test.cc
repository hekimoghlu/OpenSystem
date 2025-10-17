/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "net/dcsctp/packet/bounded_byte_reader.h"

#include "api/array_view.h"
#include "rtc_base/buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(BoundedByteReaderTest, CanLoadData) {
  uint8_t data[14] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};

  BoundedByteReader<8> reader(data);
  EXPECT_EQ(reader.variable_data_size(), 6U);
  EXPECT_EQ(reader.Load32<0>(), 0x01020304U);
  EXPECT_EQ(reader.Load32<4>(), 0x05060708U);
  EXPECT_EQ(reader.Load16<4>(), 0x0506U);
  EXPECT_EQ(reader.Load8<4>(), 0x05U);
  EXPECT_EQ(reader.Load8<5>(), 0x06U);

  BoundedByteReader<6> sub = reader.sub_reader<6>(0);
  EXPECT_EQ(sub.Load16<0>(), 0x0900U);
  EXPECT_EQ(sub.Load32<0>(), 0x09000102U);
  EXPECT_EQ(sub.Load16<4>(), 0x0304U);

  EXPECT_THAT(reader.variable_data(), ElementsAre(9, 0, 1, 2, 3, 4));
}

}  // namespace
}  // namespace dcsctp
