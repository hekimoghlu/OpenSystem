/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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

#ifdef HAVE_LINUX_AUDIT

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <libaudit.h>

#include "sudoers.h"
#include "linux_audit.h"

#define AUDIT_NOT_CONFIGURED	-2

/*
 * Open audit connection if possible.
 * Returns audit fd on success and -1 on failure.
 */
static int
linux_audit_open(void)
{
    static int au_fd = -1;
    debug_decl(linux_audit_open, SUDOERS_DEBUG_AUDIT);

    if (au_fd != -1)
	debug_return_int(au_fd);
    au_fd = audit_open();
    if (au_fd == -1) {
	/* Kernel may not have audit support. */
	if (errno == EINVAL || errno == EPROTONOSUPPORT || errno == EAFNOSUPPORT)
	    au_fd = AUDIT_NOT_CONFIGURED;
	else
	    sudo_warn("%s", U_("unable to open audit system"));
    } else if (fcntl(au_fd, F_SETFD, FD_CLOEXEC) == -1) {
	sudo_warn("%s", U_("unable to open audit system"));
	audit_close(au_fd);
	au_fd = -1;
    }
    debug_return_int(au_fd);
}

int
linux_audit_command(char *const argv[], int result)
{
    int au_fd, rc = -1;
    char *cp, *command = NULL;
    char * const *av;
    size_t size, n;
    debug_decl(linux_audit_command, SUDOERS_DEBUG_AUDIT);

    /* Don't return an error if auditing is not configured. */
    if ((au_fd = linux_audit_open()) < 0)
	debug_return_int(au_fd == AUDIT_NOT_CONFIGURED ? 0 : -1);

    /* Convert argv to a flat string. */
    for (size = 0, av = argv; *av != NULL; av++)
	size += strlen(*av) + 1;
    if (size != 0)
	command = malloc(size);
    if (command == NULL) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	goto done;
    }
    for (av = argv, cp = command; *av != NULL; av++) {
	n = strlcpy(cp, *av, size - (cp - command));
	if (n >= size - (cp - command)) {
	    sudo_warnx(U_("internal error, %s overflow"), __func__);
	    goto done;
	}
	cp += n;
	*cp++ = ' ';
    }
    *--cp = '\0';

    /* Log command, ignoring ECONNREFUSED on error. */
    if (audit_log_user_command(au_fd, AUDIT_USER_CMD, command, NULL, result) <= 0) {
	if (errno != ECONNREFUSED) {
	    sudo_warn("%s", U_("unable to send audit message"));
	    goto done;
	}
    }

    rc = 0;

done:
    free(command);

    debug_return_int(rc);
}

#endif /* HAVE_LINUX_AUDIT */
