/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "sudoers.h"

/*
 * Similar to setenv(3) but operates on a private copy of the environment.
 * Does not include warnings or debugging to avoid recursive calls.
 */
static int
sudo_setenv_nodebug(const char *var, const char *val, int overwrite)
{
    char *ep, *estring = NULL;
    const char *cp;
    size_t esize;
    int ret = -1;

    if (var == NULL || *var == '\0') {
	errno = EINVAL;
	goto done;
    }

    /*
     * POSIX says a var name with '=' is an error but BSD
     * just ignores the '=' and anything after it.
     */
    for (cp = var; *cp && *cp != '='; cp++)
	continue;
    esize = (size_t)(cp - var) + 2;
    if (val) {
	esize += strlen(val);	/* glibc treats a NULL val as "" */
    }

    /* Allocate and fill in estring. */
    if ((estring = ep = malloc(esize)) == NULL)
	goto done;
    for (cp = var; *cp && *cp != '='; cp++)
	*ep++ = *cp;
    *ep++ = '=';
    if (val) {
	for (cp = val; *cp; cp++)
	    *ep++ = *cp;
    }
    *ep = '\0';

    ret = sudo_putenv_nodebug(estring, true, overwrite);
done:
    if (ret == -1)
	free(estring);
    else
	sudoers_gc_add(GC_PTR, estring);
    return ret;
}

int
sudoers_hook_getenv(const char *name, char **value, void *closure)
{
    static bool in_progress = false; /* avoid recursion */

    if (in_progress || env_get() == NULL)
	return SUDO_HOOK_RET_NEXT;

    in_progress = true;

    /* Hack to make GNU gettext() find the sudoers locale when needed. */
    if (*name == 'L' && sudoers_getlocale() == SUDOERS_LOCALE_SUDOERS) {
	if (strcmp(name, "LANGUAGE") == 0 || strcmp(name, "LANG") == 0) {
	    *value = NULL;
	    goto done;
	}
	if (strcmp(name, "LC_ALL") == 0 || strcmp(name, "LC_MESSAGES") == 0) {
	    *value = (char *)def_sudoers_locale;
	    goto done;
	}
    }

    *value = sudo_getenv_nodebug(name);
done:
    in_progress = false;
    return SUDO_HOOK_RET_STOP;
}

int
sudoers_hook_putenv(char *string, void *closure)
{
    static bool in_progress = false; /* avoid recursion */

    if (in_progress || env_get() == NULL)
	return SUDO_HOOK_RET_NEXT;

    in_progress = true;
    sudo_putenv_nodebug(string, true, true);
    in_progress = false;
    return SUDO_HOOK_RET_STOP;
}

int
sudoers_hook_setenv(const char *name, const char *value, int overwrite, void *closure)
{
    static bool in_progress = false; /* avoid recursion */

    if (in_progress || env_get() == NULL)
	return SUDO_HOOK_RET_NEXT;

    in_progress = true;
    sudo_setenv_nodebug(name, value, overwrite);
    in_progress = false;
    return SUDO_HOOK_RET_STOP;
}

int
sudoers_hook_unsetenv(const char *name, void *closure)
{
    static bool in_progress = false; /* avoid recursion */

    if (in_progress || env_get() == NULL)
	return SUDO_HOOK_RET_NEXT;

    in_progress = true;
    sudo_unsetenv_nodebug(name);
    in_progress = false;
    return SUDO_HOOK_RET_STOP;
}
