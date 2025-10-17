/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
 * Landon Kurt Knoll
 * Phong Vo
 *
 * FNV-1 linear congruent checksum/hash/PRNG
 * see http://www.isthe.com/chongo/tech/comp/fnv/
 */

#ifndef _FNV_H
#define _FNV_H

#include <ast_common.h>

#define FNV_INIT	0x811c9dc5L
#define FNV_MULT	0x01000193L

#define FNVINIT(h)	(h = FNV_INIT)
#define FNVPART(h,c)	(h = (h) * FNV_MULT ^ (c))
#define FNVSUM(h,s,n)	do { \
			register size_t _i_ = 0; \
			while (_i_ < n) \
				FNVPART(h, ((unsigned char*)s)[_i_++]); \
			} while (0)

#if _typ_int64_t

#ifdef _ast_LL

#define FNV_INIT64	0xcbf29ce484222325LL
#define FNV_MULT64	0x00000100000001b3LL

#else

#define FNV_INIT64	((int64_t)0xcbf29ce484222325)
#define FNV_MULT64	((int64_t)0x00000100000001b3)

#endif

#define FNVINIT64(h)	(h = FNV_INIT64)
#define FNVPART64(h,c)	(h = (h) * FNV_MULT64 ^ (c))
#define FNVSUM64(h,s,n)	do { \
			register int _i_ = 0; \
			while (_i_ < n) \
				FNVPART64(h, ((unsigned char*)s)[_i_++]); \
			} while (0)

#endif

#endif
