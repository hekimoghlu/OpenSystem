/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#ifndef MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_BURSTY_H_
#define MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_BURSTY_H_

// This file contains a set of packets masks for the FEC code. The masks in
// this table are specifically designed to favor recovery of bursty/consecutive
// loss network conditions. The tradeoff is worse recovery for random losses.
// These packet masks are currently defined to protect up to 12 media packets.
// They have the following property: for any packet mask defined by the
// parameters (k,m), where k = number of media packets, m = number of FEC
// packets, all "consecutive" losses of size <= m are completely recoverable.
// By consecutive losses we mean consecutive with respect to the sequence
// number ordering of the list (media and FEC) of packets. The difference
// between these masks (`kFecMaskBursty`) and `kFecMaskRandom` type, defined
// in fec_private_tables.h, is more significant for longer codes
// (i.e., more packets/symbols in the code, so larger (k,m), i.e.,  k > 4,
// m > 3).

#include <stdint.h>

namespace webrtc {
namespace fec_private_tables {

extern const uint8_t kPacketMaskBurstyTbl[];

}  // namespace fec_private_tables
}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_FEC_PRIVATE_TABLES_BURSTY_H_
