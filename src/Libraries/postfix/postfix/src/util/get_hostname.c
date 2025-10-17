/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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
#include <sys/param.h>
#include <string.h>
#include <unistd.h>

#if (MAXHOSTNAMELEN < 256)
#undef MAXHOSTNAMELEN
#define MAXHOSTNAMELEN	256
#endif

/* Utility library. */

#include "mymalloc.h"
#include "msg.h"
#include "valid_hostname.h"
#include "get_hostname.h"

/* Local stuff. */

static char *my_host_name;

/* get_hostname - look up my host name */

const char *get_hostname(void)
{
    char    namebuf[MAXHOSTNAMELEN + 1];

    /*
     * The gethostname() call is not (or not yet) in ANSI or POSIX, but it is
     * part of the socket interface library. We avoid the more politically-
     * correct uname() routine because that has no portable way of dealing
     * with long (FQDN) hostnames.
     * 
     * DO NOT CALL GETHOSTBYNAME FROM THIS FUNCTION. IT BREAKS MAILDIR DELIVERY
     * AND OTHER THINGS WHEN THE MACHINE NAME IS NOT FOUND IN /ETC/HOSTS OR
     * CAUSES PROCESSES TO HANG WHEN THE NETWORK IS DISCONNECTED.
     * 
     * POSTFIX NO LONGER NEEDS A FULLY QUALIFIED HOSTNAME. INSTEAD POSTFIX WILL
     * USE A DEFAULT DOMAIN NAME "LOCALDOMAIN".
     */
    if (my_host_name == 0) {
	/* DO NOT CALL GETHOSTBYNAME FROM THIS FUNCTION */
	if (gethostname(namebuf, sizeof(namebuf)) < 0)
	    msg_fatal("gethostname: %m");
	namebuf[MAXHOSTNAMELEN] = 0;
	/* DO NOT CALL GETHOSTBYNAME FROM THIS FUNCTION */
	if (valid_hostname(namebuf, DO_GRIPE) == 0)
	    msg_fatal("unable to use my own hostname");
	/* DO NOT CALL GETHOSTBYNAME FROM THIS FUNCTION */
	my_host_name = mystrdup(namebuf);
    }
    return (my_host_name);
}
