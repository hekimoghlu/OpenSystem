/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
 *  minibless.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Mon Aug 25 2003.
 *  Copyright (c) 2003-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: minibless.c,v 1.12 2006/07/18 22:09:51 ssen Exp $
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <err.h>

#include "bless.h"
#include "bless_private.h"
#include "protos.h"

void miniusage(char *program);

int main(int argc, char *argv[]) {
	
    char *mountpath = NULL;
    char *device = NULL;
    struct statfs sb;
	
    
    if(argc != 2)
		miniusage(argv[0]);
	
    mountpath = argv[1];
	
    if(0 != statfs(mountpath, &sb)) {
		err(1, "Can't access %s", mountpath);
    }
	
    device = sb.f_mntfromname;
	
	// normally i would object to using preprocessor macros for harware
	// tests on principle. in this case, we do this so that IOKit
	// features and stuff that only apply to 10.4 and greater are only
	// linked in for the i386 side, for instance if we were to use per-arch
	// SDKs in the future.
#ifdef __ppc__
    if(!BLIsOpenFirmwarePresent(NULL)) {
		errx(1, "OpenFirmware not present");
    }
	
    if(0 != BLSetOpenFirmwareBootDevice(NULL, device)) {
		errx(1, "Can't set OpenFirmware");
    }
#else
#if defined(__i386__) || defined(__x86_64__)
	if(0 != setefidevice(NULL, device + 5 /* strlen("/dev/") */,
			     0,
			     0,
			     NULL,
			     NULL,
                       false)) {
		errx(1, "Can't set EFI");		
	}
	
#else
#error wha?????
#endif
#endif
	
    return 0;
}

void miniusage(char *program)
{
    FILE *mystderr = fdopen(STDERR_FILENO, "w");
    
    if(mystderr == NULL)
		err(1, "Can't open stderr");
    
    fprintf(mystderr, "Usage: %s mountpoint\n", program);
    exit(1);
}

// we don't implement output
int blesscontextprintf(BLContextPtr context, int loglevel, char const *fmt, ...) {
	return 0;
}
