/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_LEB128_H_
#define MODULES_RTP_RTCP_SOURCE_LEB128_H_

#include <cstdint>

namespace webrtc {

// Returns number of bytes needed to store `value` in leb128 format.
int Leb128Size(uint64_t value);

// Reads leb128 encoded value and advance read_at by number of bytes consumed.
// Sets read_at to nullptr on error.
uint64_t ReadLeb128(const uint8_t*& read_at, const uint8_t* end);

// Encodes `value` in leb128 format. Assumes buffer has size of at least
// Leb128Size(value). Returns number of bytes consumed.
int WriteLeb128(uint64_t value, uint8_t* buffer);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_LEB128_H_
