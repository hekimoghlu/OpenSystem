/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#include "packer.h"
#include <string.h>
#include <stdlib.h>

static char __unused vers_id[] = "stringlib.c : v2.3p2 Alec Muffett 18 May 1993";

char
Chop(string)
    register char *string;
{
    register char c;
    register char *ptr;
    c = '\0';

    for (ptr = string; *ptr; ptr++);
    if (ptr != string)
    {
	c = *(--ptr);
	*ptr = '\0';
    }
    return (c);
}

char *
Trim(string)
    register char *string;
{
    register char *ptr;
    for (ptr = string; *ptr; ptr++);

    while ((--ptr >= string) && isspace(*ptr));

    *(++ptr) = '\0';

    return (ptr);
}

char *
Clone(string)
    char *string;
{
    register char *retval;
    int len = strlen(string)+1;
    retval = (char *) malloc(len);
    if (retval)
    {
	strlcpy(retval, string, len);
    }
    return (retval);
}
