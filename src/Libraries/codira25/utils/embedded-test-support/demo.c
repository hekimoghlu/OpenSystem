/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift open source project
//
// Copyright (c) 2024 Apple Inc. and the Swift project authors.
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://language.org/LICENSE.txt for license information
// See https://language.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stddef.h>

void *malloc(size_t count);

int global1 = 42;
int global2 = 777;

void *global_with_reloc = &global1;

__attribute__((noinline))
int recur(int p, int n) {
  if (p == 0) return 1;
  global2++;
  return recur(p - n, n) + 1;
}

__attribute__((noinline))
void *testheap() {
  void *p = malloc(12);
  *(uint32_t *)p = 1234;
  return p;
}


int puts(const char *);
int main() {
  puts("Hello Embedded Swift!\n");
  puts("-- printing works\n");
  int res = recur(10, 1);
  if (res == 11) 
    puts("-- stack works\n");
  else
    puts("???\n");

  if (global1 == 42)
    puts("-- global1 works\n");
  else
    puts("???\n");

  if (global2 == 787)
    puts("-- global2 works\n");
  else
    puts("???\n");

  if ((void *)global_with_reloc == (void *)&global1)
    puts("-- global_with_reloc works\n");
  else
    puts("???\n");

  if (*(int *)global_with_reloc == 42)
    puts("-- global_with_reloc has right value\n");
  else
    puts("???\n");

  void *p = testheap();
  if (*(uint32_t *)p == 1234)
    puts("-- heap work\n");
  else
    puts("???\n");

  puts("DONE!\n");

  return 0;
}
