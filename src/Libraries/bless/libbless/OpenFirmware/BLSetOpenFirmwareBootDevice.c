/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
 *  BLSetOpenFirmwareBootDevice.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLSetOpenFirmwareBootDevice.c,v 1.18 2006/02/20 22:49:57 ssen Exp $
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "bless.h"
#include "bless_private.h"

#define NVRAM "/usr/sbin/nvram"

int BLSetOpenFirmwareBootDevice(BLContextPtr context, const char * mntfrm) {
    char ofString[1024];
    int err;
    
    char * OFSettings[6];
    
    char bootdevice[1024];
    char bootfile[1024];
    char bootcommand[1024];
    char bootargs[1024]; // always zero out bootargs
    
    pid_t p;
    int status;
    
    OFSettings[0] = NVRAM;
    err = BLGetOpenFirmwareBootDevice(context, mntfrm, ofString, sizeof(ofString));
    if(err) {
        contextprintf(context, kBLLogLevelError,  "Can't get Open Firmware information\n" );
        return 1;
    } else {
        contextprintf(context, kBLLogLevelVerbose,  "Got OF string %s\n", ofString );
    }
    
    strlcpy(bootargs, "boot-args=", sizeof(bootargs));
    
    char oldbootargs[1024];
    char *restargs;
    FILE *pop;
    
    oldbootargs[0] = '\0';
    
    pop = popen("/usr/sbin/nvram boot-args", "r");
    if(pop) {
        
        if(NULL == fgets(oldbootargs, (int)sizeof(oldbootargs), pop)) {
            contextprintf(context, kBLLogLevelVerbose,  "Could not parse output from /usr/sbin/nvram\n" );
        }
        pclose(pop);
        
        restargs = oldbootargs;
        if(NULL != strsep(&restargs, "\t")) { // nvram must separate the name from the value with a tab
            restargs[strlen(restargs)-1] = '\0'; // remove \n
            
            err = BLPreserveBootArgs(context, restargs, bootargs+strlen(bootargs), (int)(sizeof bootargs - strlen(bootargs)));
        }
    }
    
    // set them up
    snprintf(bootdevice, sizeof(bootdevice), "boot-device=%s", ofString);
    snprintf(bootfile, sizeof(bootfile), "boot-file=");
    snprintf(bootcommand, sizeof(bootcommand), "boot-command=mac-boot");
	// bootargs initialized above, and append-to later
    
    OFSettings[1] = bootdevice;
    OFSettings[2] = bootfile;
    OFSettings[3] = bootcommand;
    OFSettings[4] = bootargs;
    OFSettings[5] = NULL;
    
    contextprintf(context, kBLLogLevelVerbose,  "OF Setings:\n" );    
    contextprintf(context, kBLLogLevelVerbose,  "\t\tprogram: %s\n", OFSettings[0] );
    contextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[1] );
    contextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[2] );
    contextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[3] );
    contextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[4] );
    
    p = fork();
    if (p == 0) {
        int ret = execv(NVRAM, OFSettings);
        if(ret == -1) {
            contextprintf(context, kBLLogLevelError,  "Could not exec %s\n", NVRAM );
        }
        _exit(1);
    }
    
    do {
        p = wait(&status);
    } while (p == -1 && errno == EINTR);
    
    if(p == -1 || status) {
        contextprintf(context, kBLLogLevelError,  "%s returned non-0 exit status\n", NVRAM );
        return 3;
    }
    
    return 0;
}
