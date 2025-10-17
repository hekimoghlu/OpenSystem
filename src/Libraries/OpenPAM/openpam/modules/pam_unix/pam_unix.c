/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <sys/param.h>

#include <pwd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif

#include <security/pam_modules.h>
#include <security/pam_appl.h>

#ifndef OPENPAM
static char password_prompt[] = "Password:";
#endif

#ifndef PAM_EXTERN
#define PAM_EXTERN
#endif

PAM_EXTERN int
pam_sm_authenticate(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{
#ifndef OPENPAM
	struct pam_conv *conv;
	struct pam_message msg;
	const struct pam_message *msgp;
	struct pam_response *resp;
#endif
	struct passwd *pwd;
	const char *user;
	char *crypt_password, *password;
	int pam_err, retry;

	(void)argc;
	(void)argv;

	/* identify user */
	if ((pam_err = pam_get_user(pamh, &user, NULL)) != PAM_SUCCESS)
		return (pam_err);
	if ((pwd = getpwnam(user)) == NULL)
		return (PAM_USER_UNKNOWN);

	/* get password */
#ifndef OPENPAM
	pam_err = pam_get_item(pamh, PAM_CONV, (const void **)&conv);
	if (pam_err != PAM_SUCCESS)
		return (PAM_SYSTEM_ERR);
	msg.msg_style = PAM_PROMPT_ECHO_OFF;
	msg.msg = password_prompt;
	msgp = &msg;
#endif
	for (retry = 0; retry < 3; ++retry) {
#ifdef OPENPAM
		pam_err = pam_get_authtok(pamh, PAM_AUTHTOK,
		    (const char **)&password, NULL);
#else
		resp = NULL;
		pam_err = (*conv->conv)(1, &msgp, &resp, conv->appdata_ptr);
		if (resp != NULL) {
			if (pam_err == PAM_SUCCESS)
				password = resp->resp;
			else
				free(resp->resp);
			free(resp);
		}
#endif
		if (pam_err == PAM_SUCCESS)
			break;
	}
	if (pam_err == PAM_CONV_ERR)
		return (pam_err);
	if (pam_err != PAM_SUCCESS)
		return (PAM_AUTH_ERR);

	/* compare passwords */
	if ((!pwd->pw_passwd[0] && (flags & PAM_DISALLOW_NULL_AUTHTOK)) ||
	    (crypt_password = crypt(password, pwd->pw_passwd)) == NULL ||
	    strcmp(crypt_password, pwd->pw_passwd) != 0)
		pam_err = PAM_AUTH_ERR;
	else
		pam_err = PAM_SUCCESS;
#ifndef OPENPAM
	free(password);
#endif
	return (pam_err);
}

PAM_EXTERN int
pam_sm_setcred(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{

	(void)pamh;
	(void)flags;
	(void)argc;
	(void)argv;
	return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{

	(void)pamh;
	(void)flags;
	(void)argc;
	(void)argv;
	return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_open_session(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{

	(void)pamh;
	(void)flags;
	(void)argc;
	(void)argv;
	return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_close_session(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{

	(void)pamh;
	(void)flags;
	(void)argc;
	(void)argv;
	return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_chauthtok(pam_handle_t *pamh, int flags,
	int argc, const char *argv[])
{

	(void)pamh;
	(void)flags;
	(void)argc;
	(void)argv;
	return (PAM_SERVICE_ERR);
}

#ifdef PAM_MODULE_ENTRY
PAM_MODULE_ENTRY("pam_unix");
#endif
