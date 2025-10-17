/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#ifndef _VA_LIST_T
#define _VA_LIST_T

#if defined(KERNEL)
#ifdef XNU_KERNEL_PRIVATE
/*
 * Xcode doesn't currently set up search paths correctly for Kernel extensions,
 * so the clang headers are not seen in the correct order to use their types.
 */
#endif
#define USE_CLANG_STDARG 0
#else
#if defined(__has_feature) && __has_feature(modules)
#define USE_CLANG_STDARG 1
#else
#define USE_CLANG_STDARG 0
#endif
#endif

#if USE_CLANG_STDARG
#define __need_va_list
#include <stdarg.h>
#undef __need_va_list
#else
#include <machine/types.h> /* __darwin_va_list */
typedef __darwin_va_list va_list;
#endif

#undef USE_CLANG_STDARG

#endif /* _VA_LIST_T */
