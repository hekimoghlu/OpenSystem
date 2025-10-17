/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <grp.h>
#include <pwd.h>
#include <membership.h>
#include <membershipPriv.h>
#include <sys/syslimits.h>

#define _PAM_EXTERN_FUNCTIONS
#include <security/pam_modules.h>
#include <security/pam_appl.h>
#include <security/openpam.h>
#include "Logging.h"

#ifdef PAM_USE_OS_LOG
PAM_DEFINE_LOG(sacl)
#define PAM_LOG PAM_LOG_sacl()
#endif

#define MODULE_NAME "pam_sacl"

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t * pamh, int flags,
			int argc, const char ** argv)
{
	const char *	service = NULL;
	const char *	username = NULL;
	bool		allow_trustacct = false;

	struct passwd *pwd = NULL;
	struct passwd pwdbuf;
	char pwbuffer[2 * PATH_MAX];

	uuid_t	user_uuid;
	int	err;
	int	ismember;

	service = openpam_get_option(pamh, "sacl_service");
	allow_trustacct = openpam_get_option(pamh, "allow_trustacct");

	if (!service) {
		_LOG_ERROR("%s: missing service option", MODULE_NAME);
		return PAM_IGNORE;
	}

	if (pam_get_user(pamh, &username, NULL) != PAM_SUCCESS ||
	    username == NULL || *username == '\0') {
        _LOG_ERROR("%s: missing username", MODULE_NAME);
		return PAM_SYSTEM_ERR;
	}
 
    _LOG_DEBUG("%s: checking if account '%s' can access service '%s'",
		    MODULE_NAME, username, service);

	/* Since computer trust accounts in OD are not user accounts, you can't
	 * add them to a SACL, so we always let them through (if the option is
	 * set). A computer trust account has a username ending in '$' and no
	 * corresponding user account (ie. no passwd entry).
	 */
	if (allow_trustacct) {
		const char * c;

		c = strrchr(username, '$');
		if (c && *(c + 1) == '\0' && getpwnam_r(username, &pwdbuf, pwbuffer, sizeof(pwbuffer), &pwd) == 0) {
			_LOG_VERBOSE("%s: allowing '%s' because it is a "
				"computer trust account",
				MODULE_NAME, username);
			return PAM_SUCCESS;
		}
	}

	/* Get the UUID. This will fail if the user is is logging in over
	 * SMB, is specifed as DOMAIN\user or user@REALM and the directory
	 * does not have the aliases we need.
	 */
	if (mbr_user_name_to_uuid(username, user_uuid)) {
		char * sacl_group;

		/* We couldn't map the user to a UID, but we only care about
		 * this if the relevant SACL groups exist.
		 */

		if (asprintf(&sacl_group, "com.apple.access_%s\n",
							service) == -1) {
			return PAM_SYSTEM_ERR;
		}

		if (getgrnam(sacl_group) == NULL &&
		    getgrnam("com.apple.access_all_services") == NULL) {

            _LOG_VERBOSE("%s: allowing '%s' "
				    "due to absence of service ACL",
				    MODULE_NAME, username);

			free(sacl_group);
			return PAM_SUCCESS;
		}

        _LOG_VERBOSE("%s: denying '%s' due to missing UUID",
			MODULE_NAME, username);

		free(sacl_group);
		return PAM_PERM_DENIED;
	}

	err = mbr_check_service_membership(user_uuid, service, &ismember);
	if (err) {
	        if (err == ENOENT) {
	                /* Service ACLs not configured. */
                _LOG_VERBOSE("%s: allowing '%s' "
				"due to unconfigured service ACLs",
				MODULE_NAME, username);
	                return PAM_SUCCESS;
	        }
	
        _LOG_VERBOSE("%s: denying '%s' "
			"due to failed service ACL check (errno=%d)",
			MODULE_NAME, username, err);

	        return PAM_PERM_DENIED;
	}
	
        if (ismember) {
            _LOG_VERBOSE("%s: allowing '%s'", MODULE_NAME, username);
		return PAM_SUCCESS;
	} else {
        _LOG_ERROR("%s: denying '%s' "
			"due to failed service ACL check",
			MODULE_NAME, username);
		return PAM_PERM_DENIED;
	}
}

#ifdef PAM_STATIC
PAM_MODULE_ENTRY(MODULE_NAME);
#endif

