/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/* System Headers */

#include <sys/types.h>
#include <err.h>
#include <errno.h>
#ifdef __APPLE__
#include <limits.h>
#endif /* __APPLE__ */
#include <pwd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Local headers */

#include "at.h"
#include "perm.h"
#include "privs.h"

/* Macros */

#define MAXUSERID 10

/* Structures and unions */

/* Function declarations */

static int check_for_user(FILE *fp,const char *name);

/* Local functions */

static int check_for_user(FILE *fp,const char *name)
{
    char *buffer;
    size_t len;
    int found = 0;

    len = strlen(name);
#ifdef __APPLE__
    if (len + 2 >= INT_MAX)
	errx(EXIT_FAILURE, "user name too long");
#endif /* __APPLE__ */
    
    if ((buffer = malloc(len+2)) == NULL)
	errx(EXIT_FAILURE, "virtual memory exhausted");

#ifdef __APPLE__
    while(fgets(buffer, (int)(len+2), fp) != NULL)
#else /* !__APPLE__ */
    while(fgets(buffer, len+2, fp) != NULL)
#endif /* __APPLE__ */
    {
	if ((strncmp(name, buffer, len) == 0) &&
	    (buffer[len] == '\n'))
	{
	    found = 1;
	    break;
	}
    }
    fclose(fp);
    free(buffer);
    return found;
}
/* Global functions */
int check_permission(void)
{
    FILE *fp;
    uid_t uid = geteuid();
    struct passwd *pentry;

    if (uid==0)
	return 1;

    if ((pentry = getpwuid(uid)) == NULL)
	err(EXIT_FAILURE, "cannot access user database");

    PRIV_START

    fp=fopen(PERM_PATH "at.allow","r");

    PRIV_END

    if (fp != NULL)
    {
	return check_for_user(fp, pentry->pw_name);
    }
    else if (errno == ENOENT)
    {

	PRIV_START

	fp=fopen(PERM_PATH "at.deny", "r");

	PRIV_END

	if (fp != NULL)
	{
	    return !check_for_user(fp, pentry->pw_name);
	}
	else if (errno != ENOENT)
	    warn("at.deny");
    }
    else
	warn("at.allow");
    return 0;
}
