/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#include <stdlib.h>

#include "../mte_utils.h"
#include "CHECK.h"

#if defined(__BIONIC__) && defined(__aarch64__)

int main(int, char**) {
  void* mte_tls_ptr = mte_tls();
  *reinterpret_cast<uintptr_t*>(mte_tls_ptr) = 1;
  int ret = is_stack_mte_on() && mte_tls_ptr != nullptr ? 0 : 1;
  printf("RAN\n");
  return ret;
}

#else

int main(int, char**) {
  printf("RAN\n");
  return 1;
}
#endif
