/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#ifndef NET_DCSCTP_COMMON_MATH_H_
#define NET_DCSCTP_COMMON_MATH_H_

namespace dcsctp {

// Rounds up `val` to the nearest value that is divisible by four. Frequently
// used to e.g. pad chunks or parameters to an even 32-bit offset.
template <typename IntType>
IntType RoundUpTo4(IntType val) {
  return (val + 3) & ~3;
}

// Similarly, rounds down `val` to the nearest value that is divisible by four.
template <typename IntType>
IntType RoundDownTo4(IntType val) {
  return val & ~3;
}

// Returns true if `val` is divisible by four.
template <typename IntType>
bool IsDivisibleBy4(IntType val) {
  return (val & 3) == 0;
}

}  // namespace dcsctp

#endif  // NET_DCSCTP_COMMON_MATH_H_
