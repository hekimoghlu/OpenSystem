/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"
#include "numpy/numpyconfig.h"

/* Disable broken MS math functions */
#if defined(_MSC_VER) || defined(__MINGW32_VERSION)
#undef HAVE_ATAN2
#undef HAVE_HYPOT
#endif

/* Safe to use ldexp and frexp for long double for MSVC builds */
#if (NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE) || defined(_MSC_VER)
    #ifdef HAVE_LDEXP
        #define HAVE_LDEXPL 1
    #endif
    #ifdef HAVE_FREXP
        #define HAVE_FREXPL 1
    #endif
#endif

/* Disable broken Sun Workshop Pro math functions */
#ifdef __SUNPRO_C
#undef HAVE_ATAN2
#endif

#endif
