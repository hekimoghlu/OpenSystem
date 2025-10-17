/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/public/types.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(OutgoingSSNResetRequestParameterTest, SerializeAndDeserialize) {
  OutgoingSSNResetRequestParameter parameter(
      ReconfigRequestSN(1), ReconfigRequestSN(2), TSN(3),
      {StreamID(4), StreamID(5), StreamID(6)});

  std::vector<uint8_t> serialized;
  parameter.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      OutgoingSSNResetRequestParameter deserialized,
      OutgoingSSNResetRequestParameter::Parse(serialized));

  EXPECT_EQ(*deserialized.request_sequence_number(), 1u);
  EXPECT_EQ(*deserialized.response_sequence_number(), 2u);
  EXPECT_EQ(*deserialized.sender_last_assigned_tsn(), 3u);
  EXPECT_THAT(deserialized.stream_ids(),
              ElementsAre(StreamID(4), StreamID(5), StreamID(6)));
}

}  // namespace
}  // namespace dcsctp
