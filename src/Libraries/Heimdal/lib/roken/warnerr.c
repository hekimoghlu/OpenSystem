/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include <config.h>

#include "roken.h"
#include "err.h"

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rk_warnerr(int doerrno, const char *fmt, va_list ap)
{
    int sverrno = errno;
    const char *progname = getprogname();

    if(progname != NULL){
	fprintf(stderr, "%s", progname);
	if(fmt != NULL || doerrno)
	    fprintf(stderr, ": ");
    }
    if (fmt != NULL){
	vfprintf(stderr, fmt, ap);
	if(doerrno)
	    fprintf(stderr, ": ");
    }
    if(doerrno)
	fprintf(stderr, "%s", strerror(sverrno));
    fprintf(stderr, "\n");
}
