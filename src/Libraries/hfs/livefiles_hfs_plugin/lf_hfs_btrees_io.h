/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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

//
//  lf_hfs_btrees_io.h
//  livefiles_hfs
//
//  Created by Yakov Ben Zaken on 22/03/2018.
//

#ifndef lf_hfs_btrees_io_h
#define lf_hfs_btrees_io_h

#include <stdio.h>


#include "lf_hfs.h"
#include "lf_hfs_btrees_internal.h"

/* BTree accessor routines */
OSStatus SetBTreeBlockSize(FileReference vp, ByteCount blockSize,
                                  ItemCount minBlockCount);

OSStatus GetBTreeBlock(FileReference vp, uint64_t blockNum,
                              GetBlockOptions options, BlockDescriptor *block);

OSStatus ReleaseBTreeBlock(FileReference vp, BlockDescPtr blockPtr,
                                  ReleaseBlockOptions options);

OSStatus ExtendBTreeFile(FileReference vp, FSSize minEOF, FSSize maxEOF);

void ModifyBlockStart(FileReference vp, BlockDescPtr blockPtr);

int hfs_create_attr_btree(struct hfsmount *hfsmp, u_int32_t nodesize, u_int32_t nodecnt);

u_int16_t get_btree_nodesize(struct vnode *vp);

#endif /* lf_hfs_btrees_io_h */
