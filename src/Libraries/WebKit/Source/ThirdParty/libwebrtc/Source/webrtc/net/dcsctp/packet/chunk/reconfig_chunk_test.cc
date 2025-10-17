/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "net/dcsctp/packet/chunk/reconfig_chunk.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;
using ::testing::SizeIs;

TEST(ReConfigChunkTest, FromCapture) {
  /*
  RE_CONFIG chunk
      Chunk type: RE_CONFIG (130)
      Chunk flags: 0x00
      Chunk length: 22
      Outgoing SSN reset request parameter
          Parameter type: Outgoing SSN reset request (0x000d)
          Parameter length: 18
          Re-configuration request sequence number: 2270550051
          Re-configuration response sequence number: 1905748638
          Senders last assigned TSN: 2270550066
          Stream Identifier: 6
      Chunk padding: 0000
  */

  uint8_t data[] = {0x82, 0x00, 0x00, 0x16, 0x00, 0x0d, 0x00, 0x12,
                    0x87, 0x55, 0xd8, 0x23, 0x71, 0x97, 0x6a, 0x9e,
                    0x87, 0x55, 0xd8, 0x32, 0x00, 0x06, 0x00, 0x00};

  ASSERT_HAS_VALUE_AND_ASSIGN(ReConfigChunk chunk, ReConfigChunk::Parse(data));

  const Parameters& parameters = chunk.parameters();
  EXPECT_THAT(parameters.descriptors(), SizeIs(1));
  ParameterDescriptor desc = parameters.descriptors()[0];
  ASSERT_EQ(desc.type, OutgoingSSNResetRequestParameter::kType);
  ASSERT_HAS_VALUE_AND_ASSIGN(
      OutgoingSSNResetRequestParameter req,
      OutgoingSSNResetRequestParameter::Parse(desc.data));
  EXPECT_EQ(*req.request_sequence_number(), 2270550051u);
  EXPECT_EQ(*req.response_sequence_number(), 1905748638u);
  EXPECT_EQ(*req.sender_last_assigned_tsn(), 2270550066u);
  EXPECT_THAT(req.stream_ids(), ElementsAre(StreamID(6)));
}

TEST(ReConfigChunkTest, SerializeAndDeserialize) {
  Parameters::Builder params_builder =
      Parameters::Builder().Add(OutgoingSSNResetRequestParameter(
          ReconfigRequestSN(123), ReconfigRequestSN(456), TSN(789),
          {StreamID(42), StreamID(43)}));

  ReConfigChunk chunk(params_builder.Build());

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(ReConfigChunk deserialized,
                              ReConfigChunk::Parse(serialized));

  const Parameters& parameters = deserialized.parameters();
  EXPECT_THAT(parameters.descriptors(), SizeIs(1));
  ParameterDescriptor desc = parameters.descriptors()[0];
  ASSERT_EQ(desc.type, OutgoingSSNResetRequestParameter::kType);
  ASSERT_HAS_VALUE_AND_ASSIGN(
      OutgoingSSNResetRequestParameter req,
      OutgoingSSNResetRequestParameter::Parse(desc.data));
  EXPECT_EQ(*req.request_sequence_number(), 123u);
  EXPECT_EQ(*req.response_sequence_number(), 456u);
  EXPECT_EQ(*req.sender_last_assigned_tsn(), 789u);
  EXPECT_THAT(req.stream_ids(), ElementsAre(StreamID(42), StreamID(43)));

  EXPECT_EQ(deserialized.ToString(), "RE-CONFIG");
}

}  // namespace
}  // namespace dcsctp
