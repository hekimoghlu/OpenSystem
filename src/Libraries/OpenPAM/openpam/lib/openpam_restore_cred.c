/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
 * Restore credentials
 */

int
openpam_restore_cred(pam_handle_t *pamh)
{
	const struct pam_saved_cred *scred;
	const void *scredp;
	int r;

	ENTER();
	r = pam_get_data(pamh, PAM_SAVED_CRED, &scredp);
	if (r != PAM_SUCCESS)
		RETURNC(r);
	if (scredp == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	scred = scredp;
	if (scred->euid != geteuid()) {
		if (seteuid(scred->euid) < 0 ||
		    setgroups(scred->ngroups, scred->groups) < 0 ||
		    setegid(scred->egid) < 0)
			RETURNC(PAM_SYSTEM_ERR);
	}
	pam_set_data(pamh, PAM_SAVED_CRED, NULL, NULL);
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	=pam_get_data
 *	PAM_SYSTEM_ERR
 */

/**
 * The =openpam_restore_cred function restores the credentials saved by
 * =openpam_borrow_cred.
 *
 * >setegid 2
 * >seteuid 2
 * >setgroups 2
 */
