/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include <stdint.h>

#include "CHECK.h"

struct AlignedVar {
  int field;
  char buffer[0x1000 - sizeof(int)];
} __attribute__((aligned(0x400)));

struct SmallVar {
  int field;
  char buffer[0xeee - sizeof(int)];
};

// The single .tdata section should have a size that isn't a multiple of its
// alignment.
__thread struct AlignedVar var1 = {13};
__thread struct AlignedVar var2 = {17};
__thread struct SmallVar var3 = {19};

static uintptr_t var_addr(void* value) {
  // Maybe the optimizer would assume that the variable has the alignment it is
  // declared with.
  asm volatile("" : "+r,m"(value) : : "memory");
  return reinterpret_cast<uintptr_t>(value);
}

int main() {
  CHECK((var_addr(&var1) & 0x3ff) == 0);
  CHECK((var_addr(&var2) & 0x3ff) == 0);
  CHECK(var1.field == 13);
  CHECK(var2.field == 17);
  CHECK(var3.field == 19);
  return 0;
}
