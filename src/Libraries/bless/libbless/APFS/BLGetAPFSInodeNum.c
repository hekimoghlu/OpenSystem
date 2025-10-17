/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
/*
 *  BLGetAPFSInodeNum.c
 *  bless
 *
 *  Created by Jon Becker on 9/21/16.
 *  Copyright (c) 2001-2016 Apple Inc. All Rights Reserved.
 *
 */

#include <unistd.h>
#include <sys/attr.h>

#include "bless.h"
#include "bless_private.h"


int BLGetAPFSInodeNum(BLContextPtr context, const char *path, uint64_t *inum)
{
    int err;
    struct attrlist blist;
    struct objectinfobuf {
        uint32_t info_length;
        uint64_t fileID;
    } __attribute__((aligned(4), packed)) attrbuf;
    
    
    // Get System Folder dirID
    blist.bitmapcount = 5;
    blist.reserved = 0;
    blist.commonattr = ATTR_CMN_FILEID;
    blist.volattr = 0;
    blist.dirattr = 0;
    blist.fileattr = 0;
    blist.forkattr = 0;
    
    err = getattrlist(path, &blist, &attrbuf, sizeof(attrbuf), 0);
    if (err) {
        return errno;
    };
    *inum = attrbuf.fileID;
    return 0;
}
