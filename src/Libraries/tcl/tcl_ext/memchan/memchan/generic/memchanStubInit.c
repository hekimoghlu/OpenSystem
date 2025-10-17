/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "memchan.h"

/* !BEGIN!: Do not edit below this line. */

MemchanStubs memchanStubs = {
    TCL_STUB_MAGIC,
    NULL,
    Memchan_Init, /* 0 */
    Memchan_SafeInit, /* 1 */
    Memchan_CreateMemoryChannel, /* 2 */
    Memchan_CreateFifoChannel, /* 3 */
    Memchan_CreateFifo2Channel, /* 4 */
    Memchan_CreateZeroChannel, /* 5 */
    Memchan_CreateNullChannel, /* 6 */
    Memchan_CreateRandomChannel, /* 7 */
};

/* !END!: Do not edit above this line. */
