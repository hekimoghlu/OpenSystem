/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
 *  BLGetFileID.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLGetFileID.c,v 1.15 2006/02/20 22:49:55 ssen Exp $
 *
 */

#include <unistd.h> 
#include <errno.h>
#include <sys/attr.h>

#include "bless.h"
#include "bless_private.h"

int BLGetFileID(BLContextPtr context, const char *path, uint32_t *folderID) {

    int err;

    struct attrlist blist;
    struct objectinfobuf {
        uint32_t info_length;
        fsobj_id_t dirid;
    } attrbuf;


    // Get System Folder dirID
    blist.bitmapcount = 5;
    blist.reserved = 0;
    blist.commonattr = ATTR_CMN_OBJID;
    blist.volattr = 0;
    blist.dirattr = 0;
    blist.fileattr = 0;
    blist.forkattr = 0;

    err = getattrlist(path, &blist, &attrbuf, sizeof(attrbuf), 0);
    if (err) {
        return errno;
    };

    /*
     * the OBJID is an attribute stored in the in-core vnode in host
     * endianness. The kernel has already swapped it when loading the
     * Catalog entry from disk, so we don't need to do any swapping
     */
    
    *folderID = attrbuf.dirid.fid_objno;
    return 0;
}

