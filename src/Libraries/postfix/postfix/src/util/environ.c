/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#include "sys_defs.h"

#ifdef MISSING_SETENV_PUTENV

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern char **environ;

static int addenv(char *);		/* append entry to environment */
static int allocated = 0;		/* environ is, or is not, allocated */

#define DO_CLOBBER      1

/* namelength - determine length of name in "name=whatever" */

static ssize_t namelength(const char *name)
{
    char   *equal;

    equal = strchr(name, '=');
    return ((equal == 0) ? strlen(name) : (equal - name));
}

/* findenv - given name, locate name=value */

static char **findenv(const char *name, ssize_t len)
{
    char  **envp;

    for (envp = environ; envp && *envp; envp++)
	if (strncmp(name, *envp, len) == 0 && (*envp)[len] == '=')
	    return (envp);
    return (0);
}

#if 0

/* getenv - given name, locate value */

char   *getenv(const char *name)
{
    ssize_t len = namelength(name);
    char  **envp = findenv(name, len);

    return (envp ? *envp + len + 1 : 0);
}

/* putenv - update or append environment (name,value) pair */

int     putenv(const char *nameval)
{
    char   *equal = strchr(nameval, '=');
    char   *value = (equal ? equal : "");

    return (setenv(nameval, value, DO_CLOBBER));
}

/* unsetenv - remove variable from environment */

void    unsetenv(const char *name)
{
    char  **envp;

    while ((envp = findenv(name, namelength(name))) != 0)
	while (envp[0] = envp[1])
	    envp++;
}

#endif

/* setenv - update or append environment (name,value) pair */

int     setenv(const char *name, const char *value, int clobber)
{
    char   *destination;
    char  **envp;
    ssize_t l_name;			/* length of name part */
    unsigned int l_nameval;		/* length of name=value */

    /* Permit name= and =value. */

    l_name = namelength(name);
    envp = findenv(name, l_name);
    if (envp != 0 && clobber == 0)
	return (0);
    if (*value == '=')
	value++;
    l_nameval = l_name + strlen(value) + 1;

    /*
     * Use available memory if the old value is long enough. Never free an
     * old name=value entry because it may not be allocated.
     */

    destination = (envp != 0 && strlen(*envp) >= l_nameval) ?
	*envp : malloc(l_nameval + 1);
    if (destination == 0)
	return (-1);
    strncpy(destination, name, l_name);
    destination[l_name] = '=';
    strcpy(destination + l_name + 1, value);
    return ((envp == 0) ? addenv(destination) : (*envp = destination, 0));
}

/* cmalloc - malloc and copy block of memory */

static char *cmalloc(ssize_t new_len, char *old, ssize_t old_len)
{
    char   *new = malloc(new_len);

    if (new != 0)
	memcpy(new, old, old_len);
    return (new);
}

/* addenv - append environment entry */

static int addenv(char *nameval)
{
    char  **envp;
    ssize_t n_used;			/* number of environment entries */
    ssize_t l_used;			/* bytes used excl. terminator */
    ssize_t l_need;			/* bytes needed incl. terminator */

    for (envp = environ; envp && *envp; envp++)
	 /* void */ ;
    n_used = envp - environ;
    l_used = n_used * sizeof(*envp);
    l_need = l_used + 2 * sizeof(*envp);

    envp = allocated ?
	(char **) realloc((char *) environ, l_need) :
	(char **) cmalloc(l_need, (char *) environ, l_used);
    if (envp == 0) {
	return (-1);
    } else {
	allocated = 1;
	environ = envp;
	environ[n_used++] = nameval;		/* add new entry */
	environ[n_used] = 0;			/* terminate list */
	return (0);
    }
}

#endif
