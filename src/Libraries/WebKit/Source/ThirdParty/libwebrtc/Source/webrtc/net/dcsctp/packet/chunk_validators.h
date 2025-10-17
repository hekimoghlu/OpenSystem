/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_VALIDATORS_H_
#define NET_DCSCTP_PACKET_CHUNK_VALIDATORS_H_

#include "net/dcsctp/packet/chunk/sack_chunk.h"

namespace dcsctp {
// Validates and cleans SCTP chunks.
class ChunkValidators {
 public:
  // Given a SackChunk, will return `true` if it's valid, and `false` if not.
  static bool Validate(const SackChunk& sack);

  // Given a SackChunk, it will return a cleaned and validated variant of it.
  // RFC4960 doesn't say anything about validity of SACKs or if the Gap ACK
  // blocks must be sorted, and non-overlapping. While they always are in
  // well-behaving implementations, this can't be relied on.
  //
  // This method internally calls `Validate`, which means that you can always
  // pass a SackChunk to this method (valid or not), and use the results.
  static SackChunk Clean(SackChunk&& sack);
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_VALIDATORS_H_
