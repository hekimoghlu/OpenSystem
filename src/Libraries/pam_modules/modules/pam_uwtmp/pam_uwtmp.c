/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <utmpx.h>
#include <string.h>
#include <stdlib.h>

#define PAM_SM_SESSION
#include <security/pam_modules.h>
#include <security/pam_appl.h>
#include <security/openpam.h>
#include "Logging.h"

#ifdef PAM_USE_OS_LOG
PAM_DEFINE_LOG(uwtmp)
#define PAM_LOG PAM_LOG_uwtmp()
#endif

#define DATA_NAME "pam_uwtmp.utmpx"

struct pam_uwtmp_data {
	struct utmpx	utmpx;
	struct utmpx	backup;
	int				restore;
};

PAM_EXTERN int
populate_struct(pam_handle_t *pamh, struct utmpx *u, int populate)
{
	int status;
	char *tty;
	char *user;
	char *remhost;

	if (NULL == u)
		return PAM_SYSTEM_ERR;

	if (PAM_SUCCESS != (status = pam_get_item(pamh, PAM_USER, (const void **)&user))) {
		_LOG_ERROR("Unable to obtain the username.");
		return status;
	}
	if (NULL != user)
		strlcpy(u->ut_user, user, sizeof(u->ut_user));

	if (populate) {
		if (PAM_SUCCESS != (status = pam_get_item(pamh, PAM_TTY, (const void **)&tty))) {
            _LOG_ERROR("Unable to obtain the tty.");
			return status;
		}
		if (NULL == tty) {
            _LOG_ERROR("The tty is NULL.");
			return PAM_IGNORE;
		} else
			strlcpy(u->ut_line, tty, sizeof(u->ut_line));

		if (PAM_SUCCESS != (status = pam_get_item(pamh, PAM_RHOST, (const void **)&remhost))) {
            _LOG_ERROR("Unable to obtain the rhost.");
			return status;
		}
		if (NULL != remhost)
			strlcpy(u->ut_host, remhost, sizeof(u->ut_host));
	}

	u->ut_pid = getpid();
	gettimeofday(&u->ut_tv, NULL);

	return status;
}

PAM_EXTERN int
pam_sm_open_session(pam_handle_t *pamh, int flags, int argc, const char **argv)
{
	int status;
	struct pam_uwtmp_data *pam_data = NULL;
	struct utmpx *u = NULL;
	struct utmpx *t = NULL;
	char *tty;

	if( (pam_data = calloc(1, sizeof(*pam_data))) == NULL ) {
		_LOG_ERROR("Memory allocation error.");
		return PAM_BUF_ERR;
	}

	u = &pam_data->utmpx;
	
	// Existing utmpx entry for current terminal?
	status = pam_get_item(pamh, PAM_TTY, (const void **) &tty);
	if (status == PAM_SUCCESS && tty != NULL) {
		strlcpy(u->ut_line, tty, sizeof(u->ut_line));
		t = getutxline(u);
	}

	if (t) {
		// YES: backup existing utmpx entry + update
		_LOG_DEBUG("Updating existing entry for %s", u->ut_line);
		memcpy(&pam_data->utmpx,  t, sizeof(*t));
		memcpy(&pam_data->backup, t, sizeof(*t));
		pam_data->restore = 1;

		if (PAM_SUCCESS != (status = populate_struct(pamh, &pam_data->utmpx, 0)))
			goto err;
	} else {
		// NO: create new utmpx entry
		_LOG_DEBUG("New entry for %s", tty ?: "-");
		if (PAM_SUCCESS != (status = populate_struct(pamh, u, 1)))
			goto err;

		u->ut_type = UTMPX_AUTOFILL_MASK | USER_PROCESS;
	}

	if (PAM_SUCCESS != (status = pam_set_data(pamh, DATA_NAME, pam_data, openpam_free_data))) {
		_LOG_ERROR("There was an error setting data in the context.");
		goto err;
	}
	pam_data = NULL;

	if( pututxline(u) == NULL ) {
		_LOG_ERROR("Unable to write the utmp record.");
		status = PAM_SYSTEM_ERR;
		goto err;
	}

	return PAM_SUCCESS;

err:
	pam_set_data(pamh, DATA_NAME, NULL, NULL);
	free(pam_data);
	return status;
}

PAM_EXTERN int
pam_sm_close_session(pam_handle_t *pamh, int flags, int argc, const char **argv)
{
	int status;
	struct pam_uwtmp_data *pam_data = NULL;
	struct utmpx *u = NULL;
	int free_u = 0;

	status = pam_get_data(pamh, DATA_NAME, (const void **)&pam_data);
	if( status != PAM_SUCCESS ) {
		_LOG_DEBUG("Unable to obtain the tmp record from the context.");
	}

	if (NULL == pam_data) {
		if( (u = calloc(1, sizeof(*u))) == NULL ) {
			_LOG_ERROR("Memory allocation error.");
			return PAM_BUF_ERR;
		}
		free_u = 1;

		if (PAM_SUCCESS != (status = populate_struct(pamh, u, 1)))
			goto fin;
	} else {
		u = &pam_data->utmpx;
	}

	if (pam_data != NULL && pam_data->restore) {
		u = &pam_data->backup;
		_LOG_DEBUG("Restoring previous entry for %s", u->ut_line);
	} else {
		_LOG_VERBOSE("Dead process");
		u->ut_type = UTMPX_AUTOFILL_MASK | DEAD_PROCESS;
	}

	if( pututxline(u) == NULL ) {
		_LOG_ERROR("Unable to write the utmp record.");
		status = PAM_SYSTEM_ERR;
		goto fin;
	}

	status = PAM_SUCCESS;

fin:
	if (1 == free_u)
		free(u);
	return status;
}
