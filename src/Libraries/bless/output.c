/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
 *  output.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Wed Feb 20 2002.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: output.c,v 1.18 2005/02/03 00:42:22 ssen Exp $
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <sys/types.h>
#include "enums.h"
#include "structs.h"
#include "bless.h"
#include "protos.h"

int blesslog(void *context, int loglevel, const char *string) {
    int ret = 0;
    int willprint = 0;
    FILE *out = NULL;

    struct blesscon *con = (struct blesscon *)context;

    switch(loglevel) {
    case kBLLogLevelNormal:
        if(con->quiet) {
            willprint = 0;
        } else {
            willprint = 1;
        }
        out = stdout;
        break;
    case kBLLogLevelVerbose:
        if(con->quiet) {
            willprint = 0;
        } else if(con->verbose) {
            willprint = 1;
        } else {
            willprint = 0;
        }
        out = stderr;
        break;
    case kBLLogLevelError:
        willprint = 1;
        out = stderr;
        break;
    }

    if(willprint) {
        ret = fputs(string, out);
    }
    return ret;
}
