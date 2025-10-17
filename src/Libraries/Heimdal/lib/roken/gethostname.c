/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

#ifndef HAVE_GETHOSTNAME

#ifdef HAVE_SYS_UTSNAME_H
#include <sys/utsname.h>
#endif

/*
 * Return the local host's name in "name", up to "namelen" characters.
 * "name" will be null-terminated if "namelen" is big enough.
 * The return code is 0 on success, -1 on failure.  (The calling
 * interface is identical to gethostname(2).)
 */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
gethostname(char *name, int namelen)
{
#if defined(HAVE_UNAME)
    {
	struct utsname utsname;
	int ret;

	ret = uname (&utsname);
	if (ret < 0)
	    return ret;
	strlcpy (name, utsname.nodename, namelen);
	return 0;
    }
#else
    strlcpy (name, "some.random.host", namelen);
    return 0;
#endif
}

#endif /* GETHOSTNAME */
