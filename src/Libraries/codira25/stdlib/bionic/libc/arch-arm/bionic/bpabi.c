/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
extern long long __divdi3(long long, long long);
extern unsigned long long __udivdi3(unsigned long long, unsigned long long);

long long __gnu_ldivmod_helper(long long a, long long b, long long* remainder) {
  long long quotient = __divdi3(a, b);
  *remainder = a - b * quotient;
  return quotient;
}

unsigned long long __gnu_uldivmod_helper(unsigned long long a, unsigned long long b,
                                         unsigned long long* remainder) {
  unsigned long long quotient = __udivdi3(a, b);
  *remainder = a - b * quotient;
  return quotient;
}
