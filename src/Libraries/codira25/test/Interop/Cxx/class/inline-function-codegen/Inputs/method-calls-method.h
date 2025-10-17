/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#ifndef TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_METHOD_H
#define TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_METHOD_H

struct Incrementor {
  int increment(int t) { return t + 1; }
};

struct IncrementUser {
  int callIncrement(int value) { return Incrementor().increment(value); }
};

inline int callMethod(int value) {
  return IncrementUser().callIncrement(value);
}

#endif // TEST_INTEROP_CXX_CLASS_INLINE_FUNCTION_THROUGH_MEMBER_INPUTS_METHOD_CALLS_METHOD_H
