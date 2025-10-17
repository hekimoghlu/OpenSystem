/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#include "net/dcsctp/packet/chunk/error_chunk.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/error_cause/unrecognized_chunk_type_cause.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(ErrorChunkTest, FromCapture) {
  /*
   ERROR chunk
      Chunk type: ERROR (9)
      Chunk flags: 0x00
      Chunk length: 12
      Unrecognized chunk type cause (Type: 73 (unknown))
  */

  uint8_t data[] = {0x09, 0x00, 0x00, 0x0c, 0x00, 0x06,
                    0x00, 0x08, 0x49, 0x00, 0x00, 0x04};

  ASSERT_HAS_VALUE_AND_ASSIGN(ErrorChunk chunk, ErrorChunk::Parse(data));

  ASSERT_HAS_VALUE_AND_ASSIGN(
      UnrecognizedChunkTypeCause cause,
      chunk.error_causes().get<UnrecognizedChunkTypeCause>());

  EXPECT_THAT(cause.unrecognized_chunk(), ElementsAre(0x49, 0x00, 0x00, 0x04));
}

TEST(ErrorChunkTest, SerializeAndDeserialize) {
  ErrorChunk chunk(Parameters::Builder()
                       .Add(UnrecognizedChunkTypeCause({1, 2, 3, 4}))
                       .Build());

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(ErrorChunk deserialized,
                              ErrorChunk::Parse(serialized));
  ASSERT_HAS_VALUE_AND_ASSIGN(
      UnrecognizedChunkTypeCause cause,
      deserialized.error_causes().get<UnrecognizedChunkTypeCause>());

  EXPECT_THAT(cause.unrecognized_chunk(), ElementsAre(1, 2, 3, 4));
}

}  // namespace
}  // namespace dcsctp
