/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include <security/pam_modules.h>
#include <security/pam_appl.h>
#include <pwd.h>
#include <sys/syslimits.h>
#include "Logging.h"

#ifdef PAM_USE_OS_LOG
PAM_DEFINE_LOG(self)
#define PAM_LOG PAM_LOG_self()
#endif

PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv)
{
	const char *user = NULL, *ruser = NULL;
	struct passwd *pwd, *rpwd;
	struct passwd pwdbuf;
	char pwbuffer[2 * PATH_MAX];
	uid_t uid, ruid;

	/* get target account */
	if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS ||
	    NULL == user || 0 != getpwnam_r(user, &pwdbuf, pwbuffer, sizeof(pwbuffer), &pwd) || NULL == pwd) {
		_LOG_ERROR("Invalid user.");
		return PAM_AUTH_ERR;
	}
	uid = pwd->pw_uid;

	/* get applicant */
	if (pam_get_item(pamh, PAM_RUSER, (const void **)&ruser) != PAM_SUCCESS ||
	    NULL == ruser || 0 != getpwnam_r(ruser, &pwdbuf, pwbuffer, sizeof(pwbuffer), &rpwd) || NULL == rpwd) {
        _LOG_ERROR("Invalid remote user.");
		return PAM_AUTH_ERR;
	}
	ruid = rpwd->pw_uid;

	/* compare */
	if (uid != ruid) {
        _LOG_ERROR("The provided user and remote user do not match.");
		return PAM_AUTH_ERR;
	}

	return PAM_SUCCESS;
}
