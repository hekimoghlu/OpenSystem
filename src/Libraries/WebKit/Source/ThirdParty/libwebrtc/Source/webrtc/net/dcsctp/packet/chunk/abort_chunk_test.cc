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
#include "net/dcsctp/packet/chunk/abort_chunk.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/error_cause/user_initiated_abort_cause.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(AbortChunkTest, FromCapture) {
  /*
  ABORT chunk
      Chunk type: ABORT (6)
      Chunk flags: 0x00
      Chunk length: 8
      User initiated ABORT cause
  */

  uint8_t data[] = {0x06, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x04};

  ASSERT_HAS_VALUE_AND_ASSIGN(AbortChunk chunk, AbortChunk::Parse(data));

  ASSERT_HAS_VALUE_AND_ASSIGN(
      UserInitiatedAbortCause cause,
      chunk.error_causes().get<UserInitiatedAbortCause>());

  EXPECT_EQ(cause.upper_layer_abort_reason(), "");
}

TEST(AbortChunkTest, SerializeAndDeserialize) {
  AbortChunk chunk(/*filled_in_verification_tag=*/true,
                   Parameters::Builder()
                       .Add(UserInitiatedAbortCause("Close called"))
                       .Build());

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(AbortChunk deserialized,
                              AbortChunk::Parse(serialized));
  ASSERT_HAS_VALUE_AND_ASSIGN(
      UserInitiatedAbortCause cause,
      deserialized.error_causes().get<UserInitiatedAbortCause>());

  EXPECT_EQ(cause.upper_layer_abort_reason(), "Close called");
}

// Validates that AbortChunk doesn't make any alignment assumptions.
TEST(AbortChunkTest, SerializeAndDeserializeOneChar) {
  AbortChunk chunk(
      /*filled_in_verification_tag=*/true,
      Parameters::Builder().Add(UserInitiatedAbortCause("!")).Build());

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(AbortChunk deserialized,
                              AbortChunk::Parse(serialized));
  ASSERT_HAS_VALUE_AND_ASSIGN(
      UserInitiatedAbortCause cause,
      deserialized.error_causes().get<UserInitiatedAbortCause>());

  EXPECT_EQ(cause.upper_layer_abort_reason(), "!");
}

}  // namespace
}  // namespace dcsctp
