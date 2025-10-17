/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
 *  BLLoadFile.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 30 2002.
 *  Copyright (c) 2002-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLLoadFile.c,v 1.15 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

#include <sys/paths.h>
#include <sys/param.h>

#include "bless.h"
#include "bless_private.h"
#include "sharedUtilities.h"


int BLLoadFile(BLContextPtr context, const char * src, int useRsrcFork,
    CFDataRef* data) {

    int err = 0;
    int isHFS = 0;
    CFDataRef                output = NULL;
    CFURLRef                 loadSrc;
    char rsrcpath[MAXPATHLEN];
    CFErrorRef               error = NULL;

	if (data) *data = NULL;
    if(useRsrcFork) {
        err = BLIsMountHFS(context, src, &isHFS);
        if(err) return 2;

        if(isHFS) {
            snprintf(rsrcpath, MAXPATHLEN-1, "%s"_PATH_RSRCFORKSPEC, src);
        } else {
            strncpy(rsrcpath, src, MAXPATHLEN-1);
        }    
    } else {
        strncpy(rsrcpath, src, MAXPATHLEN-1);    
    }

    rsrcpath[MAXPATHLEN-1] = '\0';

    loadSrc = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault,
													  (UInt8 *)rsrcpath,
													  strlen(rsrcpath), 0);

    if(loadSrc == NULL) {
        return 1;
    }

    output = CreateDataFromFileURL(kCFAllocatorDefault, loadSrc, &error);
    if (error) {
        contextprintf(context, kBLLogLevelError,  "Can't load %s\n", rsrcpath );
        CFRelease(loadSrc);
        return 2;
    }

    CFRelease(loadSrc); loadSrc = NULL;

    *data = output;
    
    return 0;
}
