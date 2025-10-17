/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
 *  BLContextPrint.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Fri Apr 25 2002.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLContextPrint.c,v 1.13 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
 
#include "bless.h"
#include "bless_private.h"
 
int contextprintf(BLContextPtr context, int loglevel, char const *fmt, ...) {
    int ret;
    char *out;
    va_list ap;
    
    
    if(!context) return 0;

    
    
    if(context->version == 0 && context->logstring) {

        va_start(ap, fmt);
#if NO_VASPRINTF
	out = malloc(1024);
	ret = vsnprintf(out, 1024, fmt, ap);  
#else
	ret = vasprintf(&out, fmt, ap);  
#endif
	va_end(ap);
    
        if((ret == -1) || (out == NULL)) {
            return context->logstring(context->logrefcon, loglevel, "Memory error, log entry not available");
        }

        ret = context->logstring(context->logrefcon, loglevel, out);
        free(out);
        return ret;
    }

    return 0;
}
