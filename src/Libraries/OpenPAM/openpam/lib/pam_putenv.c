/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
 * XSSO 4.2.1
 * XSSO 6 page 56
 *
 * Set the value of an environment variable
 */

int
pam_putenv(pam_handle_t *pamh,
	const char *namevalue)
{
	char **env, *p;
	int i;

	ENTER();
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* sanity checks */
	if (namevalue == NULL || (p = strchr(namevalue, '=')) == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* see if the variable is already in the environment */
	if ((i = openpam_findenv(pamh, namevalue, p - namevalue)) >= 0) {
		if ((p = strdup(namevalue)) == NULL)
			RETURNC(PAM_BUF_ERR);
		FREE(pamh->env[i]);
		pamh->env[i] = p;
		RETURNC(PAM_SUCCESS);
	}

	/* grow the environment list if necessary */
	if (pamh->env_count == pamh->env_size) {
		env = realloc(pamh->env,
		    sizeof(char *) * (pamh->env_size * 2 + 1));
		if (env == NULL)
			RETURNC(PAM_BUF_ERR);
		pamh->env = env;
		pamh->env_size = pamh->env_size * 2 + 1;
	}

	/* add the variable at the end */
	if ((pamh->env[pamh->env_count] = strdup(namevalue)) == NULL)
		RETURNC(PAM_BUF_ERR);
	++pamh->env_count;
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =pam_putenv function sets a environment variable.
 * Its semantics are similar to those of =putenv, but it modifies the PAM
 * context's environment list instead of the application's.
 *
 * >pam_getenv
 * >pam_getenvlist
 * >pam_setenv
 * >pam_unsetenv
 */
