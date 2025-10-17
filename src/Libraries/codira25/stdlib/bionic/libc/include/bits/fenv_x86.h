/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
 * To preserve binary compatibility with FreeBSD 5.3, we pack the
 * mxcsr into some reserved fields, rather than changing sizeof(fenv_t).
 */
typedef struct {
  __uint16_t __control;
  __uint16_t __mxcsr_hi;
  __uint16_t __status;
  __uint16_t __mxcsr_lo;
  __uint32_t __tag;
  char       __other[16];
} fenv_t;

typedef __uint16_t fexcept_t;

/* Exception flags */
#define FE_INVALID    0x01
#define FE_DENORMAL   0x02
#define FE_DIVBYZERO  0x04
#define FE_OVERFLOW   0x08
#define FE_UNDERFLOW  0x10
#define FE_INEXACT    0x20
#define FE_ALL_EXCEPT (FE_DIVBYZERO | FE_DENORMAL | FE_INEXACT | \
                       FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW)

/* Rounding modes */
#define FE_TONEAREST  0x0000
#define FE_DOWNWARD   0x0400
#define FE_UPWARD     0x0800
#define FE_TOWARDZERO 0x0c00

__END_DECLS
