/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_RETURNS_LARGE_CLASS_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_RETURNS_LARGE_CLASS_H

struct LargeClass {
  long long a1 = 0;
  long long a2 = 0;
  long long a3 = 0;
  long long a4 = 0;
  long long a5 = 0;
  long long a6 = 0;
  long long a7 = 0;
  long long a8 = 0;
};

LargeClass funcReturnsLargeClass() {
  LargeClass l;
  l.a2 = 2;
  l.a6 = 6;
  return l;
}

#endif // TEST_INTEROP_CXX_CLASS_INPUTS_RETURNS_LARGE_CLASS_H
