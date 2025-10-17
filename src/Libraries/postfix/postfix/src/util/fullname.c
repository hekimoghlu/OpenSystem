/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <unistd.h>
#include <pwd.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/* Utility library. */

#include "vstring.h"
#include "safe.h"
#include "fullname.h"

/* fullname - get name of user */

const char *fullname(void)
{
    static VSTRING *result;
    char   *cp;
    int     ch;
    uid_t   uid;
    struct passwd *pwd;

    if (result == 0)
	result = vstring_alloc(10);

    /*
     * Try the environment.
     */
    if ((cp = safe_getenv("NAME")) != 0)
	return (vstring_str(vstring_strcpy(result, cp)));

    /*
     * Try the password file database.
     */
    uid = getuid();
    if ((pwd = getpwuid(uid)) == 0)
	return (0);

    /*
     * Replace all `&' characters by the login name of this user, first
     * letter capitalized. Although the full name comes from the protected
     * password file, the actual data is specified by the user so we should
     * not trust its sanity.
     */
    VSTRING_RESET(result);
    for (cp = pwd->pw_gecos; (ch = *(unsigned char *) cp) != 0; cp++) {
	if (ch == ',' || ch == ';' || ch == '%')
	    break;
	if (ch == '&') {
	    if (pwd->pw_name[0]) {
		VSTRING_ADDCH(result, TOUPPER(pwd->pw_name[0]));
		vstring_strcat(result, pwd->pw_name + 1);
	    }
	} else {
	    VSTRING_ADDCH(result, ch);
	}
    }
    VSTRING_TERMINATE(result);
    return (vstring_str(result));
}

#ifdef TEST

#include <stdio.h>

int     main(int unused_argc, char **unused_argv)
{
    const char *cp = fullname();

    printf("%s\n", cp ? cp : "null!");
    return (0);
}

#endif
