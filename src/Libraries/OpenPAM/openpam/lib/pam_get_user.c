/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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

#include <stdlib.h>

#include <security/pam_appl.h>
#include <security/openpam.h>

#include "openpam_impl.h"

static const char user_prompt[] = "Login:";

/*
 * XSSO 4.2.1
 * XSSO 6 page 52
 *
 * Retrieve user name
 */

int
pam_get_user(pam_handle_t *pamh,
	const char **user,
	const char *prompt)
{
	const void *promptp;
	char *resp;
	int r;

	ENTER();
	if (pamh == NULL || user == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	r = pam_get_item(pamh, PAM_USER, (const void **)user);
	if (r == PAM_SUCCESS && *user != NULL)
		RETURNC(PAM_SUCCESS);
	if (prompt == NULL) {
		r = pam_get_item(pamh, PAM_USER_PROMPT, &promptp);
		if (r != PAM_SUCCESS || promptp == NULL)
			prompt = user_prompt;
		else
			prompt = promptp;
	}
	r = pam_prompt(pamh, PAM_PROMPT_ECHO_ON, &resp, "%s", prompt);
	if (r != PAM_SUCCESS)
		RETURNC(r);
	r = pam_set_item(pamh, PAM_USER, resp);
	FREE(resp);
	if (r != PAM_SUCCESS)
		RETURNC(r);
	r = pam_get_item(pamh, PAM_USER, (const void **)user);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=pam_get_item
 *	=pam_prompt
 *	=pam_set_item
 *	!PAM_SYMBOL_ERR
 */

/**
 * The =pam_get_user function returns the name of the target user, as
 * specified to =pam_start.
 * If no user was specified, nor set using =pam_set_item, =pam_get_user
 * will prompt for a user name.
 * Either way, a pointer to the user name is stored in the location
 * pointed to by the =user argument.
 *
 * The =prompt argument specifies a prompt to use if no user name is
 * cached.
 * If it is =NULL, the =PAM_USER_PROMPT will be used.
 * If that item is also =NULL, a hardcoded default prompt will be used.
 *
 * >pam_get_item
 * >pam_get_authtok
 */
