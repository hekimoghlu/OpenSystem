/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include "net/dcsctp/rx/interleaved_reassembly_streams.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "net/dcsctp/common/sequence_numbers.h"
#include "net/dcsctp/packet/chunk/forward_tsn_common.h"
#include "net/dcsctp/packet/chunk/iforward_tsn_chunk.h"
#include "net/dcsctp/packet/data.h"
#include "net/dcsctp/rx/reassembly_streams.h"
#include "net/dcsctp/testing/data_generator.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Property;

class InterleavedReassemblyStreamsTest : public testing::Test {
 protected:
  UnwrappedTSN tsn(uint32_t value) { return tsn_.Unwrap(TSN(value)); }

  InterleavedReassemblyStreamsTest() {}
  DataGenerator gen_;
  UnwrappedTSN::Unwrapper tsn_;
};

TEST_F(InterleavedReassemblyStreamsTest,
       AddUnorderedMessageReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Unordered({1}, "B")), 1);
  EXPECT_EQ(streams.Add(tsn(2), gen_.Unordered({2, 3, 4})), 3);
  EXPECT_EQ(streams.Add(tsn(3), gen_.Unordered({5, 6})), 2);
  // Adding the end fragment should make it empty again.
  EXPECT_EQ(streams.Add(tsn(4), gen_.Unordered({7}, "E")), -6);
}

TEST_F(InterleavedReassemblyStreamsTest,
       AddSimpleOrderedMessageReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Ordered({1}, "B")), 1);
  EXPECT_EQ(streams.Add(tsn(2), gen_.Ordered({2, 3, 4})), 3);
  EXPECT_EQ(streams.Add(tsn(3), gen_.Ordered({5, 6})), 2);
  EXPECT_EQ(streams.Add(tsn(4), gen_.Ordered({7}, "E")), -6);
}

TEST_F(InterleavedReassemblyStreamsTest,
       AddMoreComplexOrderedMessageReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Ordered({1}, "B")), 1);
  Data late = gen_.Ordered({2, 3, 4});
  EXPECT_EQ(streams.Add(tsn(3), gen_.Ordered({5, 6})), 2);
  EXPECT_EQ(streams.Add(tsn(4), gen_.Ordered({7}, "E")), 1);

  EXPECT_EQ(streams.Add(tsn(5), gen_.Ordered({1}, "BE")), 1);
  EXPECT_EQ(streams.Add(tsn(6), gen_.Ordered({5, 6}, "B")), 2);
  EXPECT_EQ(streams.Add(tsn(7), gen_.Ordered({7}, "E")), 1);
  EXPECT_EQ(streams.Add(tsn(2), std::move(late)), -8);
}

TEST_F(InterleavedReassemblyStreamsTest,
       DeleteUnorderedMessageReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Unordered({1}, "B")), 1);
  EXPECT_EQ(streams.Add(tsn(2), gen_.Unordered({2, 3, 4})), 3);
  EXPECT_EQ(streams.Add(tsn(3), gen_.Unordered({5, 6})), 2);

  IForwardTsnChunk::SkippedStream skipped[] = {
      IForwardTsnChunk::SkippedStream(IsUnordered(true), StreamID(1), MID(0))};
  EXPECT_EQ(streams.HandleForwardTsn(tsn(3), skipped), 6u);
}

TEST_F(InterleavedReassemblyStreamsTest,
       DeleteSimpleOrderedMessageReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Ordered({1}, "B")), 1);
  EXPECT_EQ(streams.Add(tsn(2), gen_.Ordered({2, 3, 4})), 3);
  EXPECT_EQ(streams.Add(tsn(3), gen_.Ordered({5, 6})), 2);

  IForwardTsnChunk::SkippedStream skipped[] = {
      IForwardTsnChunk::SkippedStream(IsUnordered(false), StreamID(1), MID(0))};
  EXPECT_EQ(streams.HandleForwardTsn(tsn(3), skipped), 6u);
}

TEST_F(InterleavedReassemblyStreamsTest,
       DeleteManyOrderedMessagesReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Ordered({1}, "B")), 1);
  gen_.Ordered({2, 3, 4});
  EXPECT_EQ(streams.Add(tsn(3), gen_.Ordered({5, 6})), 2);
  EXPECT_EQ(streams.Add(tsn(4), gen_.Ordered({7}, "E")), 1);

  EXPECT_EQ(streams.Add(tsn(5), gen_.Ordered({1}, "BE")), 1);
  EXPECT_EQ(streams.Add(tsn(6), gen_.Ordered({5, 6}, "B")), 2);
  EXPECT_EQ(streams.Add(tsn(7), gen_.Ordered({7}, "E")), 1);

  // Expire all three messages
  IForwardTsnChunk::SkippedStream skipped[] = {
      IForwardTsnChunk::SkippedStream(IsUnordered(false), StreamID(1), MID(2))};
  EXPECT_EQ(streams.HandleForwardTsn(tsn(8), skipped), 8u);
}

TEST_F(InterleavedReassemblyStreamsTest,
       DeleteOrderedMessageDelivesTwoReturnsCorrectSize) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Ordered({1}, "B")), 1);
  gen_.Ordered({2, 3, 4});
  EXPECT_EQ(streams.Add(tsn(3), gen_.Ordered({5, 6})), 2);
  EXPECT_EQ(streams.Add(tsn(4), gen_.Ordered({7}, "E")), 1);

  EXPECT_EQ(streams.Add(tsn(5), gen_.Ordered({1}, "BE")), 1);
  EXPECT_EQ(streams.Add(tsn(6), gen_.Ordered({5, 6}, "B")), 2);
  EXPECT_EQ(streams.Add(tsn(7), gen_.Ordered({7}, "E")), 1);

  // The first ordered message expire, and the following two are delivered.
  IForwardTsnChunk::SkippedStream skipped[] = {
      IForwardTsnChunk::SkippedStream(IsUnordered(false), StreamID(1), MID(0))};
  EXPECT_EQ(streams.HandleForwardTsn(tsn(4), skipped), 8u);
}

TEST_F(InterleavedReassemblyStreamsTest, CanReassembleFastPathUnordered) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  {
    testing::InSequence s;
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(1)),
                     Property(&DcSctpMessage::payload, ElementsAre(1))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(3)),
                     Property(&DcSctpMessage::payload, ElementsAre(3))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(2)),
                     Property(&DcSctpMessage::payload, ElementsAre(2))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(4)),
                     Property(&DcSctpMessage::payload, ElementsAre(4))));
  }

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  EXPECT_EQ(streams.Add(tsn(1), gen_.Unordered({1}, "BE")), 0);
  EXPECT_EQ(streams.Add(tsn(3), gen_.Unordered({3}, "BE")), 0);
  EXPECT_EQ(streams.Add(tsn(2), gen_.Unordered({2}, "BE")), 0);
  EXPECT_EQ(streams.Add(tsn(4), gen_.Unordered({4}, "BE")), 0);
}

TEST_F(InterleavedReassemblyStreamsTest, CanReassembleFastPathOrdered) {
  NiceMock<MockFunction<ReassemblyStreams::OnAssembledMessage>> on_assembled;

  {
    testing::InSequence s;
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(1)),
                     Property(&DcSctpMessage::payload, ElementsAre(1))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(2)),
                     Property(&DcSctpMessage::payload, ElementsAre(2))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(3)),
                     Property(&DcSctpMessage::payload, ElementsAre(3))));
    EXPECT_CALL(on_assembled,
                Call(ElementsAre(tsn(4)),
                     Property(&DcSctpMessage::payload, ElementsAre(4))));
  }

  InterleavedReassemblyStreams streams("", on_assembled.AsStdFunction());

  Data data1 = gen_.Ordered({1}, "BE");
  Data data2 = gen_.Ordered({2}, "BE");
  Data data3 = gen_.Ordered({3}, "BE");
  Data data4 = gen_.Ordered({4}, "BE");
  EXPECT_EQ(streams.Add(tsn(1), std::move(data1)), 0);
  EXPECT_EQ(streams.Add(tsn(3), std::move(data3)), 1);
  EXPECT_EQ(streams.Add(tsn(2), std::move(data2)), -1);
  EXPECT_EQ(streams.Add(tsn(4), std::move(data4)), 0);
}
}  // namespace
}  // namespace dcsctp
