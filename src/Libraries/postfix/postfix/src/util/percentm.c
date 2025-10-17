/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
/* System libraries. */

#include <sys_defs.h>
#include <string.h>

/* Utility library. */

#include "vstring.h"
#include "percentm.h"

/* percentm - replace %m by error message corresponding to value in err */

char   *percentm(const char *str, int err)
{
    static VSTRING *vp;
    const unsigned char *ip = (const unsigned char *) str;

    if (vp == 0)
	vp = vstring_alloc(100);		/* grows on demand */
    VSTRING_RESET(vp);

    while (*ip) {
	switch (*ip) {
	default:
	    VSTRING_ADDCH(vp, *ip++);
	    break;
	case '%':
	    switch (ip[1]) {
	    default:				/* leave %<any> alone */
		VSTRING_ADDCH(vp, *ip++);
		/* FALLTHROUGH */
	    case '\0':				/* don't fall off end */
		VSTRING_ADDCH(vp, *ip++);
		break;
	    case 'm':				/* replace %m */
		vstring_strcat(vp, strerror(err));
		ip += 2;
		break;
	    }
	}
    }
    VSTRING_TERMINATE(vp);
    return (vstring_str(vp));
}

