/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

/*
 * Try to return what should be considered the default username or
 * NULL if we can't guess at all.
 */

ROKEN_LIB_FUNCTION const char * ROKEN_LIB_CALL
get_default_username (void)
{
    const char *user;

    user = getenv ("USER");
    if (user == NULL)
	user = getenv ("LOGNAME");
    if (user == NULL)
	user = getenv ("USERNAME");

#if defined(HAVE_GETLOGIN) && !defined(POSIX_GETLOGIN)
    if (user == NULL) {
	user = (const char *)getlogin ();
	if (user != NULL)
	    return user;
    }
#endif
#ifdef HAVE_PWD_H
    {
	uid_t uid = getuid ();
	struct passwd *pwd;

	if (user != NULL) {
	    pwd = k_getpwnam (user);
	    if (pwd != NULL && pwd->pw_uid == uid)
		return user;
	}
	pwd = k_getpwuid (uid);
	if (pwd != NULL)
	    return pwd->pw_name;
    }
#endif
#ifdef _WIN32
    /* TODO: We can call GetUserNameEx() and figure out a
       username. However, callers do not free the return value of this
       function. */
#endif

    return user;
}
