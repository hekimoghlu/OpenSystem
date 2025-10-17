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

#ifndef _NPY_NPY_FPMATH_H_
#define _NPY_NPY_FPMATH_H_

#include "npy_config.h"

#include "numpy/npy_os.h"
#include "numpy/npy_cpu.h"
#include "numpy/npy_common.h"

#ifdef NPY_OS_DARWIN
    /* This hardcoded logic is fragile, but universal builds makes it
     * difficult to detect arch-specific features */

    /* MAC OS X < 10.4 and gcc < 4 does not support proper long double, and
     * is the same as double on those platforms */
    #if NPY_BITSOF_LONGDOUBLE == NPY_BITSOF_DOUBLE
        /* This assumes that FPU and ALU have the same endianness */
        #if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
            #define HAVE_LDOUBLE_IEEE_DOUBLE_LE
        #elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN
            #define HAVE_LDOUBLE_IEEE_DOUBLE_BE
        #else
            #error Endianness undefined ?
        #endif
    #else
        #if defined(NPY_CPU_X86)
            #define HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE
        #elif defined(NPY_CPU_AMD64)
            #define HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
        #elif defined(NPY_CPU_PPC) || defined(NPY_CPU_PPC64)
            #define HAVE_LDOUBLE_IEEE_DOUBLE_16_BYTES_BE
        #endif
    #endif
#endif

#if !(defined(HAVE_LDOUBLE_IEEE_QUAD_BE) || \
      defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_16_BYTES_BE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
      defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE) || \
      defined(HAVE_LDOUBLE_DOUBLE_DOUBLE_BE))
    #error No long double representation defined
#endif

#endif
