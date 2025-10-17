/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <unistd.h>
#include <pwd.h>
#include <grp.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Implement "id -G" using sudo_getgrouplist2().
 */

int
main(int argc, char *argv[])
{
    char *username = NULL;
    GETGROUPS_T *groups = NULL;
    struct passwd *pw;
    int ch, i, ngroups;
    gid_t basegid;

    initprogname(argc > 0 ? argv[0] : "getgids");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v] [user]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    if (argc > 0)
	username = argv[0];

    if (username != NULL) {
	if ((pw = getpwnam(username)) == NULL)
	    sudo_fatalx("unknown user name %s", username);
    } else {
	if ((pw = getpwuid(getuid())) == NULL)
	    sudo_fatalx("unknown user ID %u", (unsigned int)getuid());
    }
    basegid = pw->pw_gid;
    if ((username = strdup(pw->pw_name)) == NULL)
	sudo_fatal(NULL);

    if (sudo_getgrouplist2(username, basegid, &groups, &ngroups) == -1)
	sudo_fatal("sudo_getgroulist2");

    for (i = 0; i < ngroups; i++) {
	printf("%s%u", i ? " " : "", (unsigned int)groups[i]);
    }
    putchar('\n');
    return EXIT_SUCCESS;
}
