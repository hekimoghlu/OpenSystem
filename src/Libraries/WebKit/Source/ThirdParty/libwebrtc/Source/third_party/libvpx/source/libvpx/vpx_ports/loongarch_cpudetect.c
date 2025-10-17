/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#include "./vpx_config.h"
#include "vpx_ports/loongarch.h"

#define LOONGARCH_CFG2 0x02
#define LOONGARCH_CFG2_LSX (1 << 6)
#define LOONGARCH_CFG2_LASX (1 << 7)

#if CONFIG_RUNTIME_CPU_DETECT
#if defined(__loongarch__) && defined(__linux__)
int loongarch_cpu_caps(void) {
  int reg = 0;
  int flag = 0;

  __asm__ volatile("cpucfg %0, %1 \n\t" : "+&r"(reg) : "r"(LOONGARCH_CFG2));
  if (reg & LOONGARCH_CFG2_LSX) flag |= HAS_LSX;

  if (reg & LOONGARCH_CFG2_LASX) flag |= HAS_LASX;

  return flag;
}
#else /* end __loongarch__ && __linux__ */
#error \
    "--enable-runtime-cpu-detect selected, but no CPU detection method " \
"available for your platform. Reconfigure with --disable-runtime-cpu-detect."
#endif
#else /* end CONFIG_RUNTIME_CPU_DETECT */
int loongarch_cpu_caps(void) { return 0; }
#endif
