/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

#ifndef __WCHAR_T
#define __WCHAR_T

#define __need_wchar_t
#include <stddef.h>
#undef __need_wchar_t

#endif /* __WCHAR_T */

#else

/* wchar_t is a built-in type in C++ */
#ifndef __cplusplus
#ifndef _WCHAR_T
#define _WCHAR_T
#include <machine/_types.h> /* __darwin_wchar_t */
typedef __darwin_wchar_t wchar_t;
#endif /* _WCHAR_T */
#endif /* __cplusplus */

#endif

#undef USE_CLANG_STDDEF
