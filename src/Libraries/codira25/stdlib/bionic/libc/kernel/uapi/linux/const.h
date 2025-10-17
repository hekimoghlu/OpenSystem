/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#ifndef _UAPI_LINUX_CONST_H
#define _UAPI_LINUX_CONST_H
#ifdef __ASSEMBLY__
#define _AC(X,Y) X
#define _AT(T,X) X
#else
#define __AC(X,Y) (X ##Y)
#define _AC(X,Y) __AC(X, Y)
#define _AT(T,X) ((T) (X))
#endif
#define _UL(x) (_AC(x, UL))
#define _ULL(x) (_AC(x, ULL))
#define _BITUL(x) (_UL(1) << (x))
#define _BITULL(x) (_ULL(1) << (x))
#ifndef __ASSEMBLY__
#define _BIT128(x) ((unsigned __int128) (1) << (x))
#endif
#define __ALIGN_KERNEL(x,a) __ALIGN_KERNEL_MASK(x, (__typeof__(x)) (a) - 1)
#define __ALIGN_KERNEL_MASK(x,mask) (((x) + (mask)) & ~(mask))
#define __KERNEL_DIV_ROUND_UP(n,d) (((n) + (d) - 1) / (d))
#endif
