/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
/* $FreeBSD: src/sys/msdosfs/bpb.h,v 1.7 1999/08/28 00:48:07 peter Exp $ */
/*	$NetBSD: bpb.h,v 1.7 1997/11/17 15:36:24 ws Exp $	*/

/*
 * Written by Paul Popelka (paulp@uts.amdahl.com)
 *
 * You can do anything you want with this software, just don't say you wrote
 * it, and don't remove this notice.
 *
 * This software is provided "as is".
 *
 * The author supplies this software to be publicly redistributed on the
 * understanding that the author is not responsible for the correct
 * functioning of this software in any circumstances and is not liable for
 * any damages caused by this software.
 *
 * October 1992
 */


/*
 * The following structures represent how the bpb's look on disk.  shorts
 * and longs are just character arrays of the appropriate length.  This is
 * because the compiler forces shorts and longs to align on word or
 * halfword boundaries.
 *
 * XXX The little-endian code here assumes that the processor can access
 * 16-bit and 32-bit quantities on byte boundaries.  If this is not true,
 * use the macros for the big-endian case.
 */
#include <machine/endian.h>
#if (BYTE_ORDER == LITTLE_ENDIAN) 			/* && defined(UNALIGNED_ACCESS) */
#define	getushort(x)	*((u_int16_t *)(x))
#define	getulong(x)	*((u_int32_t *)(x))
#else
#define getushort(x)	(((u_int8_t *)(x))[0] + (((u_int8_t *)(x))[1] << 8))
#define getulong(x)	(((u_int8_t *)(x))[0] + (((u_int8_t *)(x))[1] << 8) \
			 + (((u_int8_t *)(x))[2] << 16)	\
			 + (((u_int8_t *)(x))[3] << 24))
#endif

/*
 * BPB for DOS 7.10 (FAT32).  This one has a few extensions to bpb50.
 */
struct byte_bpb710 {
	u_int8_t bpbBytesPerSec[2];	/* bytes per sector */
	u_int8_t bpbSecPerClust;	/* sectors per cluster */
	u_int8_t bpbResSectors[2];	/* number of reserved sectors */
	u_int8_t bpbFATs;		/* number of FATs */
	u_int8_t bpbRootDirEnts[2];	/* number of root directory entries */
	u_int8_t bpbSectors[2];		/* total number of sectors */
	u_int8_t bpbMedia;		/* media descriptor */
	u_int8_t bpbFATsecs[2];		/* number of sectors per FAT */
	u_int8_t bpbSecPerTrack[2];	/* sectors per track */
	u_int8_t bpbHeads[2];		/* number of heads */
	u_int8_t bpbHiddenSecs[4];	/* # of hidden sectors */
	u_int8_t bpbHugeSectors[4];	/* # of sectors if bpbSectors == 0 */
	u_int8_t bpbBigFATsecs[4];	/* like bpbFATsecs for FAT32 */
	u_int8_t bpbExtFlags[2];	/* extended flags: */
	u_int8_t bpbFSVers[2];		/* filesystem version */
	u_int8_t bpbRootClust[4];	/* start cluster for root directory */
	u_int8_t bpbFSInfo[2];		/* filesystem info structure sector */
	u_int8_t bpbBackup[2];		/* backup boot sector */
	/* There is a 12 byte filler here, but we ignore it */
};

