/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
#include "net/dcsctp/packet/parameter/add_incoming_streams_request_parameter.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"

namespace dcsctp {
namespace {

TEST(AddIncomingStreamsRequestParameterTest, SerializeAndDeserialize) {
  AddIncomingStreamsRequestParameter parameter(ReconfigRequestSN(1), 2);

  std::vector<uint8_t> serialized;
  parameter.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      AddIncomingStreamsRequestParameter deserialized,
      AddIncomingStreamsRequestParameter::Parse(serialized));

  EXPECT_EQ(*deserialized.request_sequence_number(), 1u);
  EXPECT_EQ(deserialized.nbr_of_new_streams(), 2u);
}

}  // namespace
}  // namespace dcsctp
