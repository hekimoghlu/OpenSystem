/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#include "common_video/h265/h265_inline.h"

#include <stdint.h>

// Table used by WebRtcVideo_CountLeadingZeros32_NotBuiltin. For each uint32_t n
// that's a sequence of 0 bits followed by a sequence of 1 bits, the entry at
// index (n * 0x8c0b2891) >> 26 in this table gives the number of zero bits in
// n.
const int8_t kWebRtcVideo_CountLeadingZeros32_Table[64] = {
    32, 8,  17, -1, -1, 14, -1, -1, -1, 20, -1, -1, -1, 28, -1, 18,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  26, 25, 24,
    4,  11, 23, 31, 3,  7,  10, 16, 22, 30, -1, -1, 2,  6,  13, 9,
    -1, 15, -1, 21, -1, 29, 19, -1, -1, -1, -1, -1, 1,  27, 5,  12,
};
