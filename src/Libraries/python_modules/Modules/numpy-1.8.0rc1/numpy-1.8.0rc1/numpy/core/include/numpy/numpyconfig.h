/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#ifndef _NPY_NUMPYCONFIG_H_
#define _NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * harcoded
 */
#ifdef __APPLE__
    #undef NPY_SIZEOF_LONG
    #undef NPY_SIZEOF_PY_INTPTR_T

    #ifdef __LP64__
        #define NPY_SIZEOF_LONG         8
        #define NPY_SIZEOF_PY_INTPTR_T  8
    #else
        #define NPY_SIZEOF_LONG         4
        #define NPY_SIZEOF_PY_INTPTR_T  4
    #endif
#endif

/**
 * To help with the NPY_NO_DEPRECATED_API macro, we include API version
 * numbers for specific versions of NumPy. To exclude all API that was
 * deprecated as of 1.7, add the following before #including any NumPy
 * headers:
 *   #define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
 */
#define NPY_1_7_API_VERSION 0x00000007
#define NPY_1_8_API_VERSION 0x00000008

#endif
