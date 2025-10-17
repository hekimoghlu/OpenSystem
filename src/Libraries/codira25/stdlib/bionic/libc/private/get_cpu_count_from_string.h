/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#include <ctype.h>
#include <stdlib.h>

// Parse a string like: 0, 2-4, 6.
static int GetCpuCountFromString(const char* s) {
  int cpu_count = 0;
  int last_cpu = -1;
  while (*s != '\0') {
    if (isdigit(*s)) {
      int cpu = static_cast<int>(strtol(s, const_cast<char**>(&s), 10));
      if (last_cpu != -1) {
        cpu_count += cpu - last_cpu;
      } else {
        cpu_count++;
      }
      last_cpu = cpu;
    } else {
      if (*s == ',') {
        last_cpu = -1;
      }
      s++;
    }
  }
  return cpu_count;
}
