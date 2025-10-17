/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#if (!defined(HAVE_STRERROR_R) && !defined(strerror_r)) || (!defined(STRERROR_R_PROTO_COMPATIBLE) && defined(HAVE_STRERROR_R))

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "roken.h"

#ifdef _MSC_VER

int ROKEN_LIB_FUNCTION
rk_strerror_r(int eno, char * strerrbuf, size_t buflen)
{
    errno_t err;

    err = strerror_s(strerrbuf, buflen, eno);
    if (err != 0) {
        int code;
        code = sprintf_s(strerrbuf, buflen, "Error % occurred.", eno);
        err = ((code != 0)? errno : 0);
    }

    return err;
}

#else  /* _MSC_VER */

int ROKEN_LIB_FUNCTION
rk_strerror_r(int eno, char *strerrbuf, size_t buflen)
{
    /* Assume is the linux broken strerror_r (returns the a buffer (char *) if the input buffer wasn't use */
#ifdef HAVE_STRERROR_R
    const char *str;
    str = strerror_r(eno, strerrbuf, buflen);
    if (str != strerrbuf)
	if (strlcpy(strerrbuf, str, buflen) >= buflen)
	    return ERANGE;
    return 0;
#else
    int ret;
    ret = strlcpy(strerrbuf, strerror(eno), buflen);
    if (ret > buflen)
	return ERANGE;
    return 0;
#endif
}

#endif  /* !_MSC_VER */

#endif
