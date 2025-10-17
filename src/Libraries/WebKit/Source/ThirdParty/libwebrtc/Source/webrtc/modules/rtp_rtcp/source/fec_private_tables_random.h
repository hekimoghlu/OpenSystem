/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_RANDOM_H_
#define MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_RANDOM_H_

// This file contains a set of packets masks for the FEC code. The masks in
// this table are specifically designed to favor recovery to random loss.
// These packet masks are defined to protect up to maximum of 48 media packets.

#include <stdint.h>

namespace webrtc {
namespace fec_private_tables {

extern const uint8_t kPacketMaskRandomTbl[];

}  // namespace fec_private_tables
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_RANDOM_H_
