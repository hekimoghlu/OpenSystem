/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

cc -O2
DES_LONG is 'unsigned int'

options    des ecb/s
 4 r2 p    181146.14 100.0%
16 r2 p    172102.94  95.0%
 4 r2 i    165424.11  91.3%
16  c p    160468.64  88.6%
 4  c p    156653.59  86.5%
 4  c i    155245.18  85.7%
 4 r1 p    154729.68  85.4%
16 r2 i    154137.69  85.1%
16 r1 p    152357.96  84.1%
16  c i    148743.91  82.1%
 4 r1 i    146695.59  81.0%
16 r1 i    144961.00  80.0%
-DDES_RISC2 -DDES_PTR

