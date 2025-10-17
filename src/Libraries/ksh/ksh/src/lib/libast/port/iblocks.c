/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
 * aux function for <ls.h> iblocks() macro
 *
 * return number of blocks, including indirect block count
 * given stat info
 *
 * mail gsf@research.att.com when you figure out the stat.st_blocks units
 * until then we assume LS_BLOCKSIZE (512)
 */

#include <ast.h>
#if _AIX /* XXX */
#undef	major
#undef	minor
#undef	makedev
#endif
#include <ast_param.h>
#include <ls.h>

#if !_mem_st_blocks_stat

#ifndef B_DIRECT
#define B_DIRECT	10
#endif

#ifdef BITFS

#define B_SIZE		BSIZE(st->st_dev)
#define B_INDIRECT	NINDIR(st->st_dev)

#else

#ifdef BSIZE
#define B_SIZE		BSIZE
#else
#define B_SIZE		1024
#endif

#ifdef NINDIR
#define B_INDIRECT	NINDIR
#else
#define B_INDIRECT	128
#endif

#endif

#endif

off_t
_iblocks(register struct stat* st)
{
#if _mem_st_blocks_stat

	return (st->st_blocks <= 0 || st->st_size <= 0) ? 0 : st->st_blocks;

#else
	unsigned long	b;
	unsigned long	t;

	t = b = (st->st_size + B_SIZE - 1) / B_SIZE;
	if ((b -= B_DIRECT) > 0)
	{
		t += (b - 1) / B_INDIRECT + 1;
		if ((b -= B_INDIRECT) > 0)
		{
			t += (b - 1) / (B_INDIRECT * B_INDIRECT) + 1;
			if (b > B_INDIRECT * B_INDIRECT)
				t++;
		}
	}
	return t * B_SIZE / LS_BLOCKSIZE;
#endif
}
