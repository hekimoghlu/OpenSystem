/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#ifndef CC_PRIVATE_CCZP_EXTRA_H
#define CC_PRIVATE_CCZP_EXTRA_H

#include <corecrypto/cczp.h>

// operation:
//   (a * b) % mod -> r
void cczp_mul_mod(cczp_const_t zp, cc_unit* r, const cc_unit* a, const cc_unit* b);

// operation:
//   (s - t) % mod -> r
// notes:
//   automatically adjusted to account for underflow
void cczp_sub_mod(cc_size n, cc_unit* r, const cc_unit* s, const cc_unit* t, cczp_const_t mod);

// operation:
//   (s + t) % mod -> r
// notes:
//   automatically adjusted to account for overflow
void cczp_add_mod(cc_size n, cc_unit* r, const cc_unit* s, const cc_unit* t, cczp_const_t mod);

#endif // CC_PRIVATE_CCZP_EXTRA_H
