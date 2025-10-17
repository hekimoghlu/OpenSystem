/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef __misc_h__
#define __misc_h__

/*
 * Declare the function as internal to the library: the function name is
 * not exported and can't be used by a program linked to the library
 *
 * see http://gcc.gnu.org/onlinedocs/gcc-3.3.5/gcc/Function-Attributes.html#Function-Attributes
 * see http://www.nedprod.com/programs/gccvisibility.html
 */
#if defined(__GNUC__) && \
	(__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3)) || \
	defined(__SUNPRO_C) && __SUNPRO_C >= 0x590
#define INTERNAL __attribute__ ((visibility("hidden")))
#ifndef PCSC_API
#define PCSC_API __attribute__ ((visibility("default")))
#endif
#elif defined(__SUNPRO_C) && __SUNPRO_C >= 0x550
/* http://wikis.sun.com/display/SunStudio/Macros+for+Shared+Library+Symbol+Visibility */
#define INTERNAL __hidden
#define PCSC_API __global
#else
#define INTERNAL
#define PCSC_API
#endif
#define EXTERNAL PCSC_API

#if defined __GNUC__

/* GNU Compiler Collection (GCC) */
#define CONSTRUCTOR __attribute__ ((constructor))
#define DESTRUCTOR __attribute__ ((destructor))

#else

/* SUN C compiler does not use __attribute__ but #pragma init (function)
 * We can't use a # inside a #define so it is not possible to use
 * #define CONSTRUCTOR_DECLARATION(x) #pragma init (x)
 * The #pragma is used directly where needed */

/* any other */
#define CONSTRUCTOR
#define DESTRUCTOR

#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef COUNT_OF
#define COUNT_OF(arr) (sizeof(arr)/sizeof(arr[0]))
#endif

#endif /* __misc_h__ */
