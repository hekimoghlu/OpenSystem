/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

#include <sys/stat.h>

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
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>

#include "sudo_plugin.h"
#include "sudo_compat.h"

/*
 * Sample sudoers group plugin that uses an extra group file with the
 * same format as /etc/group.
 */

static sudo_printf_t sudo_log;

extern void mysetgrfile(const char *);
extern int mysetgroupent(int);
extern void myendgrent(void);
extern struct group *mygetgrnam(const char *);

static int
sample_init(int version, sudo_printf_t sudo_printf, char *const argv[])
{
    struct stat sb;

    sudo_log = sudo_printf;

    if (SUDO_API_VERSION_GET_MAJOR(version) != GROUP_API_VERSION_MAJOR) {
	sudo_log(SUDO_CONV_ERROR_MSG,
	    "group_file: incompatible major version %d, expected %d\n",
	    SUDO_API_VERSION_GET_MAJOR(version),
	    GROUP_API_VERSION_MAJOR);
	return -1;
    }

    /* Check that the group file exists and has a safe mode. */
    if (argv == NULL || argv[0] == NULL) {
	sudo_log(SUDO_CONV_ERROR_MSG,
	    "group_file: path to group file not specified\n");
	return -1;
    }
    if (stat(argv[0], &sb) != 0) {
	sudo_log(SUDO_CONV_ERROR_MSG,
	    "group_file: %s: %s\n", argv[0], strerror(errno));
	return -1;
    }
    if ((sb.st_mode & (S_IWGRP|S_IWOTH)) != 0) {
	sudo_log(SUDO_CONV_ERROR_MSG,
	    "%s must be only be writable by owner\n", argv[0]);
	return -1;
    }

    mysetgrfile(argv[0]);
    if (!mysetgroupent(1))
	return false;

    return true;
}

static void
sample_cleanup(void)
{
    myendgrent();
}

/*
 * Returns true if "user" is a member of "group", else false.
 */
static int
sample_query(const char *user, const char *group, const struct passwd *pwd)
{
    struct group *grp;
    char **member;

    grp = mygetgrnam(group);
    if (grp != NULL && grp->gr_mem != NULL) {
	for (member = grp->gr_mem; *member != NULL; member++) {
	    if (strcasecmp(user, *member) == 0)
		return true;
	}
    }

    return false;
}

sudo_dso_public struct sudoers_group_plugin group_plugin = {
    GROUP_API_VERSION,
    sample_init,
    sample_cleanup,
    sample_query
};
