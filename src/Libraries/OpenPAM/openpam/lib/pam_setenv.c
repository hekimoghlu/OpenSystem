/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include <stdio.h>
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Set the value of an environment variable
 * Mirrors setenv(3)
 */

int
pam_setenv(pam_handle_t *pamh,
	const char *name,
	const char *value,
	int overwrite)
{
	char *env;
	int r;

	ENTER();
	if (pamh == NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* sanity checks */
	if (name == NULL || value == NULL || strchr(name, '=') != NULL)
		RETURNC(PAM_SYSTEM_ERR);

	/* is it already there? */
	if (!overwrite && openpam_findenv(pamh, name, strlen(name)) >= 0)
		RETURNC(PAM_SUCCESS);

	/* set it... */
	if (asprintf(&env, "%s=%s", name, value) < 0)
		RETURNC(PAM_BUF_ERR);
	r = pam_putenv(pamh, env);
	FREE(env);
	RETURNC(r);
}

/*
 * Error codes:
 *
 *	=pam_putenv
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =pam_setenv function sets a environment variable.
 * Its semantics are similar to those of =setenv, but it modifies the PAM
 * context's environment list instead of the application's.
 *
 * >pam_getenv
 * >pam_getenvlist
 * >pam_putenv
 * >pam_unsetenv
 */
