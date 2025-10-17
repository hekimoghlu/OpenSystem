/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
 *  modeUnbless.c
 *  bless
 *
 *  Created by Shantonu Sen on 7/23/09.
 *  Copyright 2009 Apple Inc. All rights reserved.
 *
 */


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/param.h>

#include "enums.h"
#include "structs.h"

#include "bless.h"
#include "bless_private.h"
#include "protos.h"

int modeUnbless(BLContextPtr context, struct clarg actargs[klast]) {
	
    int ret;
    int isHFS;
	
    struct statfs sb;
    
	if (!actargs[kunbless].present) {
		blesscontextprintf(context, kBLLogLevelError, "No volume specified\n" );
		return 1;
	}
	
	ret = BLGetCommonMountPoint(context, actargs[kunbless].argument, "", actargs[kmount].argument);
	if(ret) {
		blesscontextprintf(context, kBLLogLevelError,  "Can't determine mount point of '%s'\n", actargs[kunbless].argument );
	} else {
		blesscontextprintf(context, kBLLogLevelVerbose,  "Mount point is '%s'\n", actargs[kmount].argument );
	}
		
    ret = BLIsMountHFS(context, actargs[kmount].argument, &isHFS);
    if(ret) {
		blesscontextprintf(context, kBLLogLevelError,  "Could not determine filesystem of %s\n", actargs[kmount].argument );
		return 1;
    }
    
    if(0 != statfs(actargs[kmount].argument, &sb)) {
        blesscontextprintf(context, kBLLogLevelError,  "Can't statfs %s\n" ,
                           actargs[kmount].argument);
        return 1;	    
    }
    
    
    if(isHFS) {
		uint32_t oldwords[8];
        
		ret = BLGetVolumeFinderInfo(context, actargs[kmount].argument, oldwords);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError,  "Error getting old Finder info words for %s\n", actargs[kmount].argument );
			return 1;
		}
		
		/* unbless! unbless */
		
		oldwords[0] = 0;
		oldwords[1] = 0;
		oldwords[2] = 0;
		oldwords[3] = 0;
		oldwords[4] = 0;
		oldwords[5] = 0;
		
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[0] = %d\n", oldwords[0] );
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[1] = %d\n", oldwords[1] );
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[2] = %d\n", oldwords[2] );
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[3] = %d\n", oldwords[3] );
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[4] = %d\n", oldwords[4] );
		blesscontextprintf(context, kBLLogLevelVerbose,  "finderinfo[5] = %d\n", oldwords[5] );
		
		
		if(geteuid() != 0 && geteuid() != sb.f_owner) {
		    blesscontextprintf(context, kBLLogLevelError,  "Authorization required\n" );
			return 1;
		}
		
		ret = BLSetVolumeFinderInfo(context,  actargs[kmount].argument, oldwords);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError,  "Can't set Finder info fields for volume mounted at %s: %s\n", actargs[kmount].argument , strerror(errno));
			return 2;
		}
		
    }
		
    return 0;
}
