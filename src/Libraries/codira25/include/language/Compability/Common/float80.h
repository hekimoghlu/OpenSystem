/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
/* This header is usable in both C and C++ code.
 * Isolates build compiler checks to determine if the 80-bit
 * floating point format is supported via a particular C type.
 * It defines CFloat80Type and CppFloat80Type aliases for this
 * C type.
 */

#ifndef LANGUAGE_COMPABILITY_COMMON_FLOAT80_H_
#define LANGUAGE_COMPABILITY_COMMON_FLOAT80_H_

#include "api-attrs.h"
#include <float.h>

#if LDBL_MANT_DIG == 64
#undef HAS_FLOAT80
#define HAS_FLOAT80 1
#endif

#if defined(RT_DEVICE_COMPILATION) && defined(__CUDACC__)
/*
 * 'long double' is treated as 'double' in the CUDA device code,
 * and there is no support for 80-bit floating point format.
 * This is probably true for most offload devices, so RT_DEVICE_COMPILATION
 * check should be enough. For the time being, guard it with __CUDACC__
 * as well.
 */
#undef HAS_FLOAT80
#endif

#if HAS_FLOAT80
typedef long double CFloat80Type;
typedef long double CppFloat80Type;
#endif

#endif /* FORTRAN_COMMON_FLOAT80_H_ */
