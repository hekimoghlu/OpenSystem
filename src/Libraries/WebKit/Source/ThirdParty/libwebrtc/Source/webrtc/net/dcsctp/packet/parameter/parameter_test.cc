/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#include "net/dcsctp/packet/parameter/parameter.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;
using ::testing::SizeIs;

TEST(ParameterTest, SerializeDeserializeParameter) {
  Parameters parameters =
      Parameters::Builder()
          .Add(OutgoingSSNResetRequestParameter(ReconfigRequestSN(123),
                                                ReconfigRequestSN(456),
                                                TSN(789), {StreamID(42)}))
          .Build();

  rtc::ArrayView<const uint8_t> serialized = parameters.data();

  ASSERT_HAS_VALUE_AND_ASSIGN(Parameters parsed, Parameters::Parse(serialized));
  auto descriptors = parsed.descriptors();
  ASSERT_THAT(descriptors, SizeIs(1));
  EXPECT_THAT(descriptors[0].type, OutgoingSSNResetRequestParameter::kType);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      OutgoingSSNResetRequestParameter parsed_param,
      OutgoingSSNResetRequestParameter::Parse(descriptors[0].data));
  EXPECT_EQ(*parsed_param.request_sequence_number(), 123u);
  EXPECT_EQ(*parsed_param.response_sequence_number(), 456u);
  EXPECT_EQ(*parsed_param.sender_last_assigned_tsn(), 789u);
  EXPECT_THAT(parsed_param.stream_ids(), ElementsAre(StreamID(42)));
}

}  // namespace
}  // namespace dcsctp
