/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
 *  BLCopyFileFromCFData.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Fri Oct 19 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <CoreFoundation/CoreFoundation.h>

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include "bless.h"
#include "bless_private.h"

int BLCopyFileFromCFData(BLContextPtr context, const CFDataRef data,
						 const char * dest, int shouldPreallocate) {
	
    int fdw;
    CFDataRef theData = data;
    ssize_t byteswritten;
	
    fstore_t preall;
    int err = 0;
	
    fdw = open(dest, O_WRONLY|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
    if(fdw == -1) {
        contextprintf(context, kBLLogLevelError,  "Error opening %s for writing\n", dest );
        return 2;
    } else {
        contextprintf(context, kBLLogLevelVerbose,  "Opened dest at %s for writing\n", dest );
    }
	
    if (shouldPreallocate > kNoPreallocate) {
		preall.fst_length = CFDataGetLength(theData);
		preall.fst_offset = 0;
		preall.fst_flags = F_ALLOCATECONTIG;
		preall.fst_posmode = F_PEOFPOSMODE;
		
		err = fcntl(fdw, F_PREALLOCATE, &preall);
		if (err == -1 && errno == ENOTSUP) {
			contextprintf(context, kBLLogLevelVerbose,  "preallocation not supported on this filesystem for %s\n", dest );
		} else if (err == -1) {
			contextprintf(context, kBLLogLevelError,  "preallocation of %s failed\n", dest );
            if (shouldPreallocate == kMustPreallocate) {
                close(fdw);
                return 3;
            }
		} else {
			contextprintf(context, kBLLogLevelVerbose,  "0x%08X bytes preallocated for %s\n", (unsigned int)preall.fst_bytesalloc, dest );
		}
    } else {
		contextprintf(context, kBLLogLevelVerbose,  "No preallocation attempted for %s\n", dest );
    }
	
    byteswritten = write(fdw, (char *)CFDataGetBytePtr(theData), CFDataGetLength(theData));
    if(byteswritten != CFDataGetLength(theData)) {
		contextprintf(context, kBLLogLevelError,  "Error while writing to %s: %s\n", dest, strerror(errno) );
		contextprintf(context, kBLLogLevelError,  "%ld bytes written\n", byteswritten );
		close(fdw);
		return 2;
    }
    contextprintf(context, kBLLogLevelVerbose,  "\n" );
	
    close(fdw);
	
    return 0;
}

