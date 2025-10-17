/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * integral representation conversion support definitions
 * supports sizeof(integral_type)<=sizeof(intmax_t)
 */

#ifndef _SWAP_H
#define _SWAP_H

#include <ast_common.h>

#define int_swap	_ast_intswap

#define SWAP_MAX	8

#define SWAPOP(n)	(((n)&int_swap)^(n))

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern void*		swapmem(int, const void*, void*, size_t);
extern intmax_t		swapget(int, const void*, int);
extern void*		swapput(int, void*, int, intmax_t);
extern int		swapop(const void*, const void*, int);

#undef	extern

#endif
