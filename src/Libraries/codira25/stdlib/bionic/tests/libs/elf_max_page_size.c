/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "elf_max_page_size.h"

const int ro0 = RO0;
const int ro1 = RO1;
int rw0 = RW0;

/* Force some padding alignment */
int rw1 __attribute__((aligned(0x10000))) = RW1;

int bss0, bss1;

int* const prw0 = &rw0;

int loader_test_func(void) {
  rw0 += RW0_INCREMENT;
  rw1 += RW1_INCREMENT;

  bss0 += BSS0_INCREMENT;
  bss1 += BSS1_INCREMENT;

  return ro0 + ro1 + rw0 + rw1 + bss0 + bss1 + *prw0;
}
