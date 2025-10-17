/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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


#include "sensitive.h"

#include <stdio.h>

struct SmallCStruct getSmallStruct(int x) {
  printf("x = %d\n", x);

  struct SmallCStruct s;
  s.a = 0xdeadbeaf;
  s.b = 0xdeadbeaf;
  s.c = 0xdeadbeaf;
  return s;
}

struct LargeCStruct getLargeStruct(int x) {
  printf("x = %d\n", x);

  struct LargeCStruct s;
  s.a = 0xdeadbeaf;
  s.b = 0xdeadbeaf;
  s.c = 0xdeadbeaf;
  s.d = 0xdeadbeaf;
  s.e = 0xdeadbeaf;
  s.f = 0xdeadbeaf;
  s.g = 0xdeadbeaf;
  return s;
}

void printSmallStruct(int x, struct SmallCStruct s, int y) {
  printf("x = %d, y = %d\n", x, y);
  printf("s = (%u, %u, %u)\n", s.a, s.b, s.c);
}

struct SmallCStruct forwardSmallStruct(struct SmallCStruct s) {
  return s;
}

void printLargeStruct(int x, struct LargeCStruct s, int y) {
  printf("x = %d, y = %d\n", x, y);
  printf("s = (%u, %u, %u, %u, %u, %u, %u)\n", s.a, s.b, s.c, s.e, s.e, s.f, s.g);
}

struct LargeCStruct forwardLargeStruct(struct LargeCStruct s) {
  return s;
}

