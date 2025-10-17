/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

#include <grp.h>
#include <limits.h>
#include <pwd.h>
#include <stdlib.h>
#include <unistd.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Temporarily borrow user credentials
 */

int
openpam_borrow_cred(pam_handle_t *pamh,
	const struct passwd *pwd)
{
	struct pam_saved_cred *scred;
	const void *scredp;
	int r;

	ENTERI(pwd->pw_uid);
	r = pam_get_data(pamh, PAM_SAVED_CRED, &scredp);
	if (r == PAM_SUCCESS && scredp != NULL) {
		openpam_log(PAM_LOG_LIBDEBUG,
		    "already operating under borrowed credentials");
		RETURNC(PAM_SYSTEM_ERR);
	}
	if (geteuid() != 0 && geteuid() != pwd->pw_uid) {
		openpam_log(PAM_LOG_LIBDEBUG, "called with non-zero euid: %d",
		    (int)geteuid());
		RETURNC(PAM_PERM_DENIED);
	}
	scred = calloc(1, sizeof *scred);
	if (scred == NULL)
		RETURNC(PAM_BUF_ERR);
	scred->euid = geteuid();
	scred->egid = getegid();
	r = getgroups(NGROUPS_MAX, scred->groups);
	if (r < 0) {
		FREE(scred);
		RETURNC(PAM_SYSTEM_ERR);
	}
	scred->ngroups = r;
	r = pam_set_data(pamh, PAM_SAVED_CRED, scred, &openpam_free_data);
	if (r != PAM_SUCCESS) {
		FREE(scred);
		RETURNC(r);
	}
	if (geteuid() == pwd->pw_uid)
		RETURNC(PAM_SUCCESS);
	if (initgroups(pwd->pw_name, pwd->pw_gid) < 0 ||
	      setegid(pwd->pw_gid) < 0 || seteuid(pwd->pw_uid) < 0) {
		openpam_restore_cred(pamh);
		RETURNC(PAM_SYSTEM_ERR);
	}
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	=pam_set_data
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 *	PAM_PERM_DENIED
 */

/**
 * The =openpam_borrow_cred function saves the current credentials and
 * switches to those of the user specified by its =pwd argument.
 * The affected credentials are the effective UID, the effective GID, and
 * the group access list.
 * The original credentials can be restored using =openpam_restore_cred.
 *
 * >setegid 2
 * >seteuid 2
 * >setgroups 2
 */
