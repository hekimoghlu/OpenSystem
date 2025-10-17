/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif /* HAVE_STRINGS_H */
#include <grp.h>

#include "sudo_compat.h"
#include "sudo_dso.h"
#include "sudo_plugin.h"
#include "sudo_util.h"

/*
 * Sudoers group plugin that does group name-based lookups using the system
 * group database functions, similar to how sudo behaved prior to 1.7.3.
 * This can be used on systems where lookups by group ID are problematic.
 */

typedef struct group * (*sysgroup_getgrnam_t)(const char *);
typedef struct group * (*sysgroup_getgrgid_t)(gid_t);
typedef void (*sysgroup_gr_delref_t)(struct group *);

static sysgroup_getgrnam_t sysgroup_getgrnam;
static sysgroup_getgrgid_t sysgroup_getgrgid;
static sysgroup_gr_delref_t sysgroup_gr_delref;
static bool need_setent;

static int
sysgroup_init(int version, sudo_printf_t plugin_printf, char *const argv[])
{
    void *handle;

    if (SUDO_API_VERSION_GET_MAJOR(version) != GROUP_API_VERSION_MAJOR) {
	plugin_printf(SUDO_CONV_ERROR_MSG,
	    "sysgroup_group: incompatible major version %d, expected %d\n",
	    SUDO_API_VERSION_GET_MAJOR(version),
	    GROUP_API_VERSION_MAJOR);
	return -1;
    }

    /* Share group cache with sudo if possible. */
    handle = sudo_dso_findsym(SUDO_DSO_DEFAULT, "sudo_getgrnam");
    if (handle != NULL) {
	sysgroup_getgrnam = (sysgroup_getgrnam_t)handle;
    } else {
	sysgroup_getgrnam = (sysgroup_getgrnam_t)getgrnam;
	need_setent = true;
    }

    handle = sudo_dso_findsym(SUDO_DSO_DEFAULT, "sudo_getgrgid");
    if (handle != NULL) {
	sysgroup_getgrgid = (sysgroup_getgrgid_t)handle;
    } else {
	sysgroup_getgrgid = (sysgroup_getgrgid_t)getgrgid;
	need_setent = true;
    }

    handle = sudo_dso_findsym(SUDO_DSO_DEFAULT, "sudo_gr_delref");
    if (handle != NULL)
	sysgroup_gr_delref = (sysgroup_gr_delref_t)handle;

    if (need_setent)
	setgrent();

    return true;
}

static void
sysgroup_cleanup(void)
{
    if (need_setent)
	endgrent();
}

/*
 * Returns true if "user" is a member of "group", else false.
 */
static int
sysgroup_query(const char *user, const char *group, const struct passwd *pwd)
{
    char **member;
    struct group *grp;

    grp = sysgroup_getgrnam(group);
    if (grp == NULL && group[0] == '#' && group[1] != '\0') {
	const char *errstr;
	gid_t gid = sudo_strtoid(group + 1, &errstr);
	if (errstr == NULL)
	    grp = sysgroup_getgrgid(gid);
    }
    if (grp != NULL) {
	if (grp->gr_mem != NULL) {
	    for (member = grp->gr_mem; *member != NULL; member++) {
		if (strcasecmp(user, *member) == 0) {
		    if (sysgroup_gr_delref)
			sysgroup_gr_delref(grp);
		    return true;
		}
	    }
	}
	if (sysgroup_gr_delref)
	    sysgroup_gr_delref(grp);
    }

    return false;
}

sudo_dso_public struct sudoers_group_plugin group_plugin = {
    GROUP_API_VERSION,
    sysgroup_init,
    sysgroup_cleanup,
    sysgroup_query
};
