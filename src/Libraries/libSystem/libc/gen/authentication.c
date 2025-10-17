/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include <sys/param.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <err.h>
#include <errno.h>
#include <grp.h>
#include <paths.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

#include "authentication.h"

int isAuthenticatedAsAdministrator(void)
{
    if (isAuthenticatedAsRoot()) {
        return 1;
    }
    // otherwise ...
    return isAuthenticatedAsAdministratorForTask(0);
}

int isAuthenticatedAsAdministratorForTask(int taskNum)
{
    int admin = 0;
    uid_t ruid;

    if (isAuthenticatedAsRoot()) {
        return 1;
    }

    ruid = getuid();

    if (ruid) {
            gid_t groups[NGROUPS_MAX];
            int   numgroups;

            /*
             * Only allow those in group taskNum group (By default admin) to authenticate.
             */
            if ((numgroups = getgroups(NGROUPS_MAX, groups)) > 0) {
                    int i;
                    gid_t admingid = 0;
                    struct group *admingroup;

                    if ((admingroup = getgrnam(groupNameForTask(taskNum))) != NULL) {
                            admingid = admingroup->gr_gid;

                            for (i = 0; i < numgroups; i++) {
                                    if (groups[i] == admingid) {
                                            admin = 1;
                                            break;
                                    }
                            }
                    }

            }
    }
    // otherwise
    return admin;
}

int isAuthenticatedAsRoot(void)
{
    if (getuid() == 0) {
        return 1;
    }
    return 0;
}

char *groupNameForTask(int taskNum)
{
    if (taskNum == 0)
        return "admin";

    return "admin";
}

