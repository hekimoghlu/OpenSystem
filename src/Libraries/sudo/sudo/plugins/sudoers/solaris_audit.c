/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

#ifdef HAVE_SOLARIS_AUDIT

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#include <bsm/adt.h>
#include <bsm/adt_event.h>

#include "sudoers.h"
#include "solaris_audit.h"

static adt_session_data_t *ah;		/* audit session handle */
static adt_event_data_t	*event;		/* event to be generated */
static char		cwd[PATH_MAX];
static char		cmdpath[PATH_MAX];

static int
adt_sudo_common(char *const argv[])
{
	int argc;

	if (adt_start_session(&ah, NULL, ADT_USE_PROC_DATA) != 0) {
		log_warning(SLOG_NO_STDERR, "adt_start_session");
		return -1;
	}
	if ((event = adt_alloc_event(ah, ADT_sudo)) == NULL) {
		log_warning(SLOG_NO_STDERR, "alloc_event");
		(void) adt_end_session(ah);
		return -1;
	}
	if ((event->adt_sudo.cwdpath = getcwd(cwd, sizeof(cwd))) == NULL) {
		log_warning(SLOG_NO_STDERR, _("unable to get current working directory"));
	}

	/* get the real executable name */
	if (user_cmnd != NULL) {
		if (strlcpy(cmdpath, (const char *)user_cmnd,
		    sizeof(cmdpath)) >= sizeof(cmdpath)) {
			log_warningx(SLOG_NO_STDERR,
			    _("truncated audit path user_cmnd: %s"),
			    user_cmnd);
		}
	} else {
		if (strlcpy(cmdpath, argv[0],
		    sizeof(cmdpath)) >= sizeof(cmdpath)) {
			log_warningx(SLOG_NO_STDERR,
			    _("truncated audit path argv[0]: %s"),
			    argv[0]);
		}
	}

	for (argc = 0; argv[argc] != NULL; argc++)
		continue;

	event->adt_sudo.cmdpath = cmdpath;
	event->adt_sudo.argc = argc - 1;
	event->adt_sudo.argv = (char **)&argv[1];
	event->adt_sudo.envp = env_get();

	return 0;
}


/*
 * Returns 0 on success or -1 on error.
 */
int
solaris_audit_success(char *const argv[])
{
	int rc = -1;

	if (adt_sudo_common(argv) != 0) {
		return -1;
	}
	if (adt_put_event(event, ADT_SUCCESS, ADT_SUCCESS) != 0) {
		log_warning(SLOG_NO_STDERR, "adt_put_event(ADT_SUCCESS)");
	} else {
		rc = 0;
	}
	adt_free_event(event);
	(void) adt_end_session(ah);

	return rc;
}

/*
 * Returns 0 on success or -1 on error.
 */
int
solaris_audit_failure(char *const argv[], const char *errmsg)
{
	int rc = -1;

	if (adt_sudo_common(argv) != 0) {
		return -1;
	}

	event->adt_sudo.errmsg = (char *)errmsg;
	if (adt_put_event(event, ADT_FAILURE, ADT_FAIL_VALUE_PROGRAM) != 0) {
		log_warning(SLOG_NO_STDERR, "adt_put_event(ADT_FAILURE)");
	} else {
		rc = 0;
	}
	adt_free_event(event);
	(void) adt_end_session(ah);

	return rc;
}

#endif /* HAVE_SOLARIS_AUDIT */
