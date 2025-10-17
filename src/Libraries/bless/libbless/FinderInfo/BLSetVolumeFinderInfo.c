/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
 *  BLSetVolumeFinderInfo.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Apr 19 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLSetVolumeFinderInfo.c,v 1.17 2006/02/20 22:49:54 ssen Exp $
 *
 */

#include <CoreFoundation/CoreFoundation.h>

#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/attr.h>

#include "bless.h"
#include "bless_private.h"

struct volinfobuf {
  uint32_t info_length;
  uint32_t finderinfo[8];
}; 


int BLSetVolumeFinderInfo(BLContextPtr context, const char *mountpoint, uint32_t *words) {

    struct attrlist alist;
    struct volinfobuf vinfo;
    int err, i;

    alist.bitmapcount = 5;
    alist.reserved = 0;
    alist.commonattr = ATTR_CMN_FNDRINFO;
    alist.volattr = ATTR_VOL_INFO;
    alist.dirattr = 0;
    alist.fileattr = 0;
    alist.forkattr = 0;

    err = getattrlist(mountpoint, &alist, &vinfo, sizeof(vinfo), 0);
    if (err) {
		int rval = errno;
		contextprintf(context, kBLLogLevelError,  "Can't volume information for %s\n", mountpoint );
		return rval;
    }

    // only let the user overwrite words 0-5 (i.e. not VSDB)
    for (i=0; i<6; i++) {
        vinfo.finderinfo[i] = CFSwapInt32HostToBig(words[i]);
    }

    err = setattrlist(mountpoint, &alist, &vinfo.finderinfo, sizeof(vinfo.finderinfo), 0);
    if (err) {
		int rval = errno;
		contextprintf(context, kBLLogLevelError,  "Error while setting volume information for %s\n", mountpoint );
		return rval;
    }

    return 0;
}

