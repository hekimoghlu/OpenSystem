/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include <sys/cdefs.h>

#include <sys/types.h>
#include <sys/syslimits.h>

#include <grp.h>
#include <pwd.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <membership.h>
#include <stdlib.h>
#include "Logging.h"
#endif /* __APPLE__ */

#define PAM_SM_AUTH

#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/openpam.h>

#ifdef PAM_USE_OS_LOG
PAM_DEFINE_LOG(group)
#define PAM_LOG PAM_LOG_group()
#endif

PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t *pamh, int flags __unused,
    int argc __unused, const char *argv[] __unused)
{
	const char *group, *user;
	const void *ruser;
#ifndef __APPLE__
	char *const *list;
#endif /* !__APPLE__ */
	struct passwd *pwd;
	struct passwd pwdbuf;
	char pwbuffer[2 * PATH_MAX];
	struct group *grp;
#ifdef __APPLE__
	char *str1, *str, *p;
	int found_group = 0;
	uuid_t u_uuid, g_uuid;
	int ismember;
#endif /* __APPLE__ */

	/* get target account */
	if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS ||
	    user == NULL || getpwnam_r(user, &pwdbuf, pwbuffer, sizeof(pwbuffer), &pwd) != 0 || pwd == NULL) {
		_LOG_ERROR("Unable to obtain the username.");
		return (PAM_AUTH_ERR);
	}
	if (pwd->pw_uid != 0 && openpam_get_option(pamh, "root_only")) {
		_LOG_DEBUG("The root_only option means root only.");
		return (PAM_IGNORE);
	}

	/* get applicant */
	if (openpam_get_option(pamh, "ruser") &&
		(pam_get_item(pamh, PAM_RUSER, &ruser) != PAM_SUCCESS || ruser == NULL || 
		 getpwnam_r(ruser, &pwdbuf, pwbuffer, sizeof(pwbuffer), &pwd) != 0 || pwd == NULL)) {
		_LOG_ERROR("Unable to obtain the remote username.");
		return (PAM_AUTH_ERR);
	}

	/* get regulating group */
	if ((group = openpam_get_option(pamh, "group")) == NULL) {
		group = "wheel";
		_LOG_DEBUG("With no group specfified, I am defaulting to wheel.");
	}
#ifdef __APPLE__
	str1 = str = strdup(group);
	while ((p = strsep(&str, ",")) != NULL) {
		if ((grp = getgrnam(p)) == NULL || grp->gr_mem == NULL)
			continue;

		/* check if the group is empty */
		if (*grp->gr_mem == NULL)
			continue;

		found_group = 1;

		/* check membership */
		if (mbr_uid_to_uuid(pwd->pw_uid, u_uuid) != 0)
			continue;
		if (mbr_gid_to_uuid(grp->gr_gid, g_uuid) != 0)
			continue;
		if (mbr_check_membership(u_uuid, g_uuid, &ismember) != 0)
			continue;
		if (ismember)
			goto found;
	}
	if (!found_group) {
		_LOG_ERROR("The specified group (%s) could not be found.", group);
		goto failed;
	}
#else /* !__APPLE__ */
	if ((grp = getgrnam(group)) == NULL || grp->gr_mem == NULL) {
		_LOG_ERROR("The specified group (%s) is NULL.", group);
		goto failed;
	}

	/* check if the group is empty */
	if (*grp->gr_mem == NULL) {
		_LOG_ERROR("The specified group (%s) is empty.", group);
		goto failed;
	}

	/* check membership */
	if (pwd->pw_gid == grp->gr_gid)
		goto found;
	for (list = grp->gr_mem; *list != NULL; ++list)
		if (strcmp(*list, pwd->pw_name) == 0)
			goto found;
#endif /* __APPLE__ */

 not_found:
	_LOG_DEBUG("The group check failed.");
#ifdef __APPLE__
	free(str1);
#endif /* __APPLE__ */
	if (openpam_get_option(pamh, "deny"))
		return (PAM_SUCCESS);
	return (PAM_AUTH_ERR);
 found:
	_LOG_DEBUG("The group check succeeded.");
#ifdef __APPLE__
	free(str1);
#endif /* __APPLE__ */
	if (openpam_get_option(pamh, "deny"))
		return (PAM_AUTH_ERR);
	return (PAM_SUCCESS);
 failed:
	if (openpam_get_option(pamh, "fail_safe"))
		goto found;
	else
		goto not_found;
}


PAM_MODULE_ENTRY("pam_group");
