/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Unset an environment variable
 * Mirrors unsetenv(3)
 */

int
pam_unsetenv(pam_handle_t *pamh, const char *name)
{
	int i;

	ENTER();
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* sanity checks */
	if (name == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* find and remove the variable from the environment */
	if ((i = openpam_findenv(pamh, name, strlen(name))) >= 0) {
		memset(pamh->env[i], 0, strlen(pamh->env[i]));
		FREE(pamh->env[i]);
		pamh->env[i] = pamh->env[pamh->env_count-1];
		pamh->env[pamh->env_count-1] = NULL;
		pamh->env_count--;
		RETURNC(PAM_SUCCESS);
	}

	RETURNC(PAM_SYSTEM_ERR);
}

/*
 * Error codes:
 *
 *	=pam_unsetenv
 *	PAM_SYSTEM_ERR
 */

/**
 * The =pam_unsetenv function unsets a environment variable.
 * Its semantics are similar to those of =unsetenv, but it modifies the PAM
 * context's environment list instead of the application's.
 *
 * >pam_getenv
 * >pam_getenvlist
 * >pam_putenv
 */
