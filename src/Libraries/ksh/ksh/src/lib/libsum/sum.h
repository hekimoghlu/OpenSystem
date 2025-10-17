/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
 * checksum library interface
 */

#ifndef _SUM_H
#define _SUM_H

#include <ast.h>

#define SUM_SIZE	(1<<0)		/* print size too		*/
#define SUM_SCALE	(1<<1)		/* traditional size scale	*/
#define SUM_TOTAL	(1<<2)		/* print totals since sumopen	*/
#define SUM_LEGACY	(1<<3)		/* legacy field widths		*/

#define _SUM_PUBLIC_	const char*	name;

typedef struct Sumdata_s
{
	uint32_t	size;
	uint32_t	num;
	void*		buf;
} Sumdata_t;

typedef struct Sum_s
{
	_SUM_PUBLIC_
#ifdef	_SUM_PRIVATE_
	_SUM_PRIVATE_
#endif
} Sum_t;

extern Sum_t*	sumopen(const char*);
extern int	suminit(Sum_t*);
extern int	sumblock(Sum_t*, const void*, size_t);
extern int	sumdone(Sum_t*);
extern int	sumdata(Sum_t*, Sumdata_t*);
extern int	sumprint(Sum_t*, Sfio_t*, int, size_t);
extern int	sumusage(Sfio_t*);
extern int	sumclose(Sum_t*);

#endif
