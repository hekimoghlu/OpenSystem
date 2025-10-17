/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include <stdlib.h>
#include <string.h>

#include "jq.h"

int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  // Creat null-terminated string
  char *null_terminated = (char *)malloc(size + 1);
  memcpy(null_terminated, (char *)data, size);
  null_terminated[size] = '\0';

  // Fuzzer entrypoint
  jq_state *jq = NULL;
  jq = jq_init();
  if (jq != NULL) {
    if (jq_compile(jq, null_terminated)) {
      jq_dump_disassembly(jq, 2);
    }
  }
  jq_teardown(&jq);

  // Free the null-terminated string
  free(null_terminated);

  return 0;
}
