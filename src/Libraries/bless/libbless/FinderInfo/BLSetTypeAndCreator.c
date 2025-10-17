/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
 *  BLSetTypeAndCreator.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLSetTypeAndCreator.c,v 1.15 2006/02/20 22:49:54 ssen Exp $
 *
 */

#include <CoreFoundation/CoreFoundation.h>

#include <sys/types.h>
#include <unistd.h>
#include <sys/attr.h>

#include "bless.h"
#include "bless_private.h"

struct fileinfobuf {
  uint32_t info_length;
  uint32_t finderinfo[8];
}; 


int BLSetTypeAndCreator(BLContextPtr context, const char * path, uint32_t type, uint32_t creator) {
    struct attrlist		alist;
    struct fileinfobuf          finfo;
    int err;

	
    alist.bitmapcount = 5;
    alist.reserved = 0;
    alist.commonattr = ATTR_CMN_FNDRINFO;
    alist.volattr = 0;
    alist.dirattr = 0;
    alist.fileattr = 0;
    alist.forkattr = 0;

    err = getattrlist(path, &alist, &finfo, sizeof(finfo), 0);
    if(err) return 1;

    finfo.finderinfo[0] = CFSwapInt32HostToBig(type);
    finfo.finderinfo[1] = CFSwapInt32HostToBig(creator);

    err = setattrlist(path, &alist, &finfo.finderinfo, sizeof(finfo.finderinfo), 0);
    if(err) return 2;

    return 0;
}




