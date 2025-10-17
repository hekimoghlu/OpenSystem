/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef _HFS_BTREEIO_H_
#define _HFS_BTREEIO_H_

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE

#include "hfs.h"
#include "BTreesInternal.h"

/* BTree accessor routines */
extern OSStatus SetBTreeBlockSize(FileReference vp, ByteCount blockSize, 
				ItemCount minBlockCount);

extern OSStatus GetBTreeBlock(FileReference vp, u_int32_t blockNum, 
				GetBlockOptions options, BlockDescriptor *block);

extern OSStatus ReleaseBTreeBlock(FileReference vp, BlockDescPtr blockPtr, 
				ReleaseBlockOptions options);

extern OSStatus ExtendBTreeFile(FileReference vp, FSSize minEOF, FSSize maxEOF);

extern void ModifyBlockStart(FileReference vp, BlockDescPtr blockPtr);

int hfs_create_attr_btree(struct hfsmount *hfsmp, u_int32_t nodesize, u_int32_t nodecnt);

u_int16_t get_btree_nodesize(struct vnode *vp);

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* ! _HFS_BTREEIO_H_ */
