/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
 *  BLPreserveBootArgs.c
 *  bless
 *
 *  Created by Shantonu Sen on 11/16/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "bless.h"
#include "bless_private.h"

static const char *remove_boot_args[] = {
	"rd",	/* rd=(enet|disk0|md0) */
	"rp",	/* rp=nfs:1.2.3.4:/foo:bar.dmg, netboot */
	"boot-uuid",	/* UUID of filesystem/partition for rooting */
	NULL
};

int BLPreserveBootArgs(BLContextPtr context,
                       const char *input,
                       char *output,
                       int outputLen)
{
    return BLPreserveBootArgsIfChanged(context, input, output, outputLen, NULL);
}

int BLPreserveBootArgsIfChanged(BLContextPtr context,
                                const char *input,
                                char *output,
                                size_t outputLen,
                                bool *outChanged)
{
    char oldbootargs[1024];
    char bootargs[1024];
    size_t bootleft=sizeof(bootargs)-1;
    char *token, *restargs;
    int firstarg=1;
    bool changed=false;
    
    strncpy(oldbootargs, input, sizeof(oldbootargs)-1);
    oldbootargs[sizeof(oldbootargs)-1] = '\0';
    
    memset(bootargs, 0, sizeof(bootargs));
    
    contextprintf(context, kBLLogLevelVerbose,  "Old boot-args: %s\n", oldbootargs);
    
    restargs = oldbootargs;
    while((token = strsep(&restargs, " ")) != NULL) {
        int shouldbesaved = 1, i;
        contextprintf(context, kBLLogLevelVerbose, "\tGot token: %s\n", token);
        for(i=0; remove_boot_args[i]; i++) {
            // see if it's something we want
            if(remove_boot_args[i][0] == '-') {
                // -v style
                if(strcmp(remove_boot_args[i], token) == 0) {
                    shouldbesaved = 0;
                    break;
                }
            } else {
                // debug= style
                size_t keylen = strlen(remove_boot_args[i]);
                if(strlen(token) >= keylen+1
                   && strncmp(remove_boot_args[i], token, keylen) == 0
                   && token[keylen] == '=') {
                    shouldbesaved = 0;
                    break;
                }
            }
        }
        
        if(!shouldbesaved) {
            changed = true;
        } else {
            // append to bootargs if it should be preserved
            if(firstarg) {
                firstarg = 0;
            } else {
                strncat(bootargs, " ", bootleft);
                bootleft--;
            }
            
            contextprintf(context, kBLLogLevelVerbose,  "\tPreserving: %s\n", token);
            strncat(bootargs, token, bootleft);
            bootleft -= strlen(token);
        }
    }
    
    strlcpy(output, bootargs, outputLen);
    if (outChanged) *outChanged = changed;
    
    return 0;
}
