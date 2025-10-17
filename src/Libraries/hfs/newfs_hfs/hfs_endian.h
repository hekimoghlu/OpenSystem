/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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

/*
 * hfs_endian.h
 *
 * This file prototypes endian swapping routines for the HFS/HFS Plus
 * volume format.
 */
#include <hfs/hfs_format.h>
#include <libkern/OSByteOrder.h>

/*********************/
/* BIG ENDIAN Macros */
/*********************/
#if BYTE_ORDER == BIG_ENDIAN

    /* HFS is always big endian, make swaps into no-ops */
    #define SWAP_BE16(__a) (__a)
    #define SWAP_BE32(__a) (__a)
    #define SWAP_BE64(__a) (__a)
    
    /* HFS is always big endian, no swapping needed */
    #define SWAP_HFSMDB(__a)
    #define SWAP_HFSPLUSVH(__a)

/************************/
/* LITTLE ENDIAN Macros */
/************************/
#elif BYTE_ORDER == LITTLE_ENDIAN

    /* HFS is always big endian, make swaps actually swap */
    #define SWAP_BE16(__a) 							OSSwapBigToHostInt16 (__a)
    #define SWAP_BE32(__a) 							OSSwapBigToHostInt32 (__a)
    #define SWAP_BE64(__a) 							OSSwapBigToHostInt64 (__a)
    
    #define SWAP_HFSMDB(__a)						hfs_swap_HFSMasterDirectoryBlock ((__a))
    #define SWAP_HFSPLUSVH(__a)						hfs_swap_HFSPlusVolumeHeader ((__a));

#else
#warning Unknown byte order
#error
#endif

#ifdef __cplusplus
extern "C" {
#endif

void hfs_swap_HFSMasterDirectoryBlock (void *buf);
void hfs_swap_HFSPlusVolumeHeader (void *buf);

#ifdef __cplusplus
}
#endif

#endif /* __HFS_FORMAT__ */
