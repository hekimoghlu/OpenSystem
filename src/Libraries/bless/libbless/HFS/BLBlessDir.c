/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
 *  BLBlessDir.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLBlessDir.c,v 1.15 2006/02/20 22:49:55 ssen Exp $
 *
 */

#include <sys/types.h>

#include "bless.h"
#include "bless_private.h"


int BLBlessDir(BLContextPtr context, const char * mountpoint,
                uint32_t dirX, uint32_t dir9) {

    int err;
    uint32_t finderinfo[8];
    
    err = BLGetVolumeFinderInfo(context, mountpoint, finderinfo);
    if(err) {
        contextprintf(context, kBLLogLevelError,  "Can't get Finder info fields for volume mounted at %s\n", mountpoint );
        return 1;
    }

    /* If either directory was not specified, the dirID
     * variables will be 0, so we can use that to initialize
     * the FI fields */

    /* Set Finder info words 3 & 5 */
    finderinfo[3] = dir9;
    finderinfo[5] = dirX;

    if(!dirX) {
      /* The 9 folder is what we really want */
      finderinfo[0] = dir9;
    } else {
      /* X */
      finderinfo[0] = dirX;
    }

    contextprintf(context, kBLLogLevelVerbose,  "finderinfo[0] = %d\n", finderinfo[0] );
    contextprintf(context, kBLLogLevelVerbose,  "finderinfo[3] = %d\n", finderinfo[3] );
    contextprintf(context, kBLLogLevelVerbose,  "finderinfo[5] = %d\n", finderinfo[5] );
    
    err = BLSetVolumeFinderInfo(context, mountpoint, finderinfo);
    if(err) {
      contextprintf(context, kBLLogLevelError,  "Can't set Finder info fields for volume mounted at %s\n", mountpoint );
      return 2;
    }

    return 0;
}

