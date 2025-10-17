/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#pragma once

#include <sys/types.h>

__BEGIN_DECLS

/*
 * The ARM Cortex-A75 registers are described here:
 *
 * AArch64:
 *  FPCR: http://infocenter.arm.com/help/topic/com.arm.doc.100403_0200_00_en/lau1442502503726.html
 *  FPSR: http://infocenter.arm.com/help/topic/com.arm.doc.100403_0200_00_en/lau1442502526288.html
 * AArch32:
 *  FPSCR: http://infocenter.arm.com/help/topic/com.arm.doc.100403_0200_00_en/lau1442504290459.html
 */

#if defined(__LP64__)
typedef struct {
  /* FPCR, Floating-point Control Register. */
  __uint32_t __control;
  /* FPSR, Floating-point Status Register. */
  __uint32_t __status;
} fenv_t;

#else
typedef __uint32_t fenv_t;
#endif

typedef __uint32_t fexcept_t;

/* Exception flags. */
#define FE_INVALID    0x01
#define FE_DIVBYZERO  0x02
#define FE_OVERFLOW   0x04
#define FE_UNDERFLOW  0x08
#define FE_INEXACT    0x10
#define FE_DENORMAL   0x80
#define FE_ALL_EXCEPT (FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW | FE_DENORMAL)

/* Rounding modes. */
#define FE_TONEAREST  0x0
#define FE_UPWARD     0x1
#define FE_DOWNWARD   0x2
#define FE_TOWARDZERO 0x3

__END_DECLS
