/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#include <stdio.h>

static volatile char array[0x10000];
volatile char* volatile oob_ptr = &array[0x111111111];

unsigned char get_tag(__attribute__((unused)) volatile void* ptr) {
#if defined(__aarch64__)
  return static_cast<unsigned char>(reinterpret_cast<uintptr_t>(ptr) >> 56) & 0xf;
#else   // !defined(__aarch64__)
  return 0;
#endif  // defined(__aarch64__)
}

int main() {
  printf("Program loaded successfully. %p %p. ", array, oob_ptr);
  if (get_tag(array) != get_tag(oob_ptr)) {
    printf("Tags are mismatched!\n");
    return 1;
  }
  if (get_tag(array) == 0) {
    printf("Tags are zero!\n");
  } else {
    printf("Tags are non-zero\n");
  }
  return 0;
}
