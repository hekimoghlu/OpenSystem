/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#ifndef CC_PRIVATE_CCN_EXTRA_H
#define CC_PRIVATE_CCN_EXTRA_H

#include <corecrypto/ccn.h>

// operation:
//   operand / divisor -> quotient
// sizing:
//   unitlen quotient == n_operand
//   unitlen operand == n_operand
//   unitlen divisor == n_divisor
// notes:
//   this is a fast division algorithm based on Knuth's Algorithm D
void ccn_div_long(cc_size n_operand, cc_unit* quotient, const cc_unit* operand, cc_size n_divisor, const cc_unit* divisor);

// operation:
//   find `result` such that:
//     (result * a) % b == 1
// sizing:
//   unitlen a == n
//   unitlen b == n
//   unitlen result == n
int ccn_modular_multiplicative_inverse(cc_size n, cc_unit* result, const cc_unit* a, const cc_unit* b);

#endif // CC_PRIVATE_CCN_EXTRA_H
