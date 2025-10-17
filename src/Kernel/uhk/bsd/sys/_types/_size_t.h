/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#if defined(KERNEL)
#ifdef XNU_KERNEL_PRIVATE
/*
 * Xcode doesn't currently set up search paths correctly for Kernel extensions,
 * so the clang headers are not seen in the correct order to use their types.
 */
#endif
#define USE_CLANG_STDDEF 0
#else
#if defined(__has_feature) && __has_feature(modules)
#define USE_CLANG_STDDEF 1
#else
#define USE_CLANG_STDDEF 0
#endif
#endif

#if USE_CLANG_STDDEF

#ifndef __SIZE_T
#define __SIZE_T

#define __need_size_t
#include <stddef.h>
#undef __need_size_t

#endif  /* __SIZE_T */

#else

#ifndef _SIZE_T
#define _SIZE_T
#include <machine/_types.h> /* __darwin_size_t */
typedef __darwin_size_t        size_t;
#endif  /* _SIZE_T */

#endif

#if KERNEL
#if !defined(_SIZE_UT) && !defined(VM_UNSAFE_TYPES)
#define _SIZE_UT
typedef __typeof__(sizeof(int)) size_ut;
#endif /*  !defined(_SIZE_UT) && !defined(VM_UNSAFE_TYPES) */
#endif /* KERNEL */

#undef USE_CLANG_STDDEF
