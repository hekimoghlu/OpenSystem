/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#ifndef __HFS_ENDIAN_H__
#define __HFS_ENDIAN_H__

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE
/*
 * hfs_endian.h
 *
 * This file prototypes endian swapping routines for the HFS/HFS Plus
 * volume format.
 */
#include "hfs.h"
#include "BTreesInternal.h"
#include <libkern/OSByteOrder.h>

/*********************/
/* BIG ENDIAN Macros */
/*********************/
#define SWAP_BE16(__a) 							OSSwapBigToHostInt16 (__a)
#define SWAP_BE32(__a) 							OSSwapBigToHostInt32 (__a)
#define SWAP_BE64(__a) 							OSSwapBigToHostInt64 (__a)

#if BYTE_ORDER == BIG_ENDIAN
    
    /* HFS is always big endian, no swapping needed */
    #define SWAP_HFS_PLUS_FORK_DATA(__a)

/************************/
/* LITTLE ENDIAN Macros */
/************************/
#elif BYTE_ORDER == LITTLE_ENDIAN

    #define SWAP_HFS_PLUS_FORK_DATA(__a)			hfs_swap_HFSPlusForkData ((__a))

#else
#warning Unknown byte order
#error
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Constants for the "unswap" argument to hfs_swap_BTNode:
 */
enum HFSBTSwapDirection {
	kSwapBTNodeBigToHost		=	0,
	kSwapBTNodeHostToBig		=	1,

	/*
	 * kSwapBTNodeHeaderRecordOnly is used to swap just the header record
	 * of a header node from big endian (on disk) to host endian (in memory).
	 * It does not swap the node descriptor (forward/backward links, record
	 * count, etc.).  It assumes the header record is at offset 0x000E.
	 *
	 * Since HFS Plus doesn't have fixed B-tree node sizes, we have to read
	 * the header record to determine the actual node size for that tree
	 * before we can set up the B-tree control block.  We read it initially
	 * as 512 bytes, then re-read it once we know the correct node size.  Since
	 * we may not have read the entire header node the first time, we can't
	 * swap the record offsets, other records, or do most sanity checks.
	 */
	kSwapBTNodeHeaderRecordOnly	=	3
};

int  hfs_swap_BTNode (BlockDescriptor *src, vnode_t vp, enum HFSBTSwapDirection direction, 
	u_int8_t allow_empty_node);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* __HFS_FORMAT__ */
