/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
/* System libraries */

#include <sys_defs.h>
#include <string.h>

/* Utility library. */

#include "split_at.h"

/* split_at - break string at first delimiter, return remainder */

char   *split_at(char *string, int delimiter)
{
    char   *cp;

    if ((cp = strchr(string, delimiter)) != 0)
	*cp++ = 0;
    return (cp);
}

/* split_at_right - break string at last delimiter, return remainder */

char   *split_at_right(char *string, int delimiter)
{
    char   *cp;

    if ((cp = strrchr(string, delimiter)) != 0)
	*cp++ = 0;
    return (cp);
}
