/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
//
// These functions should be deprectaed!
// Try to find a better way instead of using them.
//

#ifndef _TOOL_ERRORS_H_
#define _TOOL_ERRORS_H_

#include <stdarg.h>
#include <stdio.h>
#include "SecurityTool/sharedTool/SecurityTool.h"

static const char *
sec_errstr(int err)
{
    const char *errString;
    static char buffer[12];
    
    snprintf(buffer, sizeof(buffer), "%d", err);
    errString = buffer;
    return errString;
}

static void
sec_error(const char *msg, ...) __attribute__((format(printf, 1, 2)));

static void
sec_error(const char *msg, ...)
{
    va_list args;
    
    fprintf(stderr, "%s: ", getprogname());
    
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    
    fprintf(stderr, "\n");
}

static inline void
sec_perror(const char *msg, int err)
{
    sec_error("%s: %s", msg, sec_errstr(err));
}



#endif
