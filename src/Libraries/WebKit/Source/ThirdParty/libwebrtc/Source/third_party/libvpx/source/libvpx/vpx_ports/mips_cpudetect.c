/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include <stdio.h>
#include <string.h>
#include "./vpx_config.h"
#include "vpx_ports/mips.h"

#if CONFIG_RUNTIME_CPU_DETECT
#if defined(__mips__) && defined(__linux__)
int mips_cpu_caps(void) {
  char cpuinfo_line[512];
  int flag = 0x0;
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) {
    // Assume nothing if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return 0;
  }
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line) - 1, f)) {
    if (memcmp(cpuinfo_line, "cpu model", 9) == 0) {
      // Workaround early kernel without mmi in ASEs line.
      if (strstr(cpuinfo_line, "Loongson-3")) {
        flag |= HAS_MMI;
      } else if (strstr(cpuinfo_line, "Loongson-2K")) {
        flag |= HAS_MMI | HAS_MSA;
      }
    }
    if (memcmp(cpuinfo_line, "ASEs implemented", 16) == 0) {
      if (strstr(cpuinfo_line, "loongson-mmi") &&
          strstr(cpuinfo_line, "loongson-ext")) {
        flag |= HAS_MMI;
      }
      if (strstr(cpuinfo_line, "msa")) {
        flag |= HAS_MSA;
      }
      // ASEs is the last line, so we can break here.
      break;
    }
  }
  fclose(f);
  return flag;
}
#else /* end __mips__ && __linux__ */
#error \
    "--enable-runtime-cpu-detect selected, but no CPU detection method " \
"available for your platform. Reconfigure with --disable-runtime-cpu-detect."
#endif
#else /* end CONFIG_RUNTIME_CPU_DETECT */
int mips_cpu_caps(void) { return 0; }
#endif
