/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#include <string.h>

#include "roken.h"

#ifndef HAVE_STRSEP_COPY

/* strsep, but with const stringp, so return string in buf */

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
strsep_copy(const char **stringp, const char *delim, char *buf, size_t len)
{
    const char *save = *stringp;
    size_t l;
    if(save == NULL)
	return -1;
    *stringp = *stringp + strcspn(*stringp, delim);
    l = min(len, (size_t)(*stringp - save));
    if(len > 0) {
	memcpy(buf, save, l);
	buf[l] = '\0';
    }

    l = *stringp - save;
    if(**stringp == '\0')
	*stringp = NULL;
    else
	(*stringp)++;
    return l;
}

#endif
