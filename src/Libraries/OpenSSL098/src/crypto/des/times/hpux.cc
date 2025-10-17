/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

HPUX 10 - 9000/887 - cc -D_HPUX_SOURCE -Aa +ESlit +O2 -Wl,-a,archive

options    des ecb/s
16  c i    149448.90 100.0%
 4  c i    145861.79  97.6%
16 r2 i    141710.96  94.8%
16 r1 i    139455.33  93.3%
 4 r2 i    138800.00  92.9%
 4 r1 i    136692.65  91.5%
16 r2 p    110228.17  73.8%
16 r1 p    109397.07  73.2%
16  c p    109209.89  73.1%
 4  c p    108014.71  72.3%
 4 r2 p    107873.88  72.2%
 4 r1 p    107685.83  72.1%
-DDES_UNROLL

