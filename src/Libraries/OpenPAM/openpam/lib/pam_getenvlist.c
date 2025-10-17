/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
 * XSSO 6 page 45
 *
 * Returns a list of all the PAM environment variables
 */

char **
pam_getenvlist(pam_handle_t *pamh)
{
	char **envlist;
	int i;

	ENTER();
	if (pamh == NULL)
		RETURNP(NULL);
	envlist = malloc(sizeof(char *) * (pamh->env_count + 1));
	if (envlist == NULL) {
		openpam_log(PAM_LOG_ERROR, "%s",
			pam_strerror(pamh, PAM_BUF_ERR));
		RETURNP(NULL);
	}
	for (i = 0; i < pamh->env_count; ++i) {
		if ((envlist[i] = strdup(pamh->env[i])) == NULL) {
			while (i) {
				--i;
				FREE(envlist[i]);
			}
			FREE(envlist);
			openpam_log(PAM_LOG_ERROR, "%s",
				pam_strerror(pamh, PAM_BUF_ERR));
			RETURNP(NULL);
		}
	}
	envlist[i] = NULL;
	RETURNP(envlist);
}

/**
 * The =pam_getenvlist function returns a copy of the given PAM context's
 * environment list as a pointer to an array of strings.
 * The last element in the array is =NULL.
 * The pointer is suitable for assignment to {Va environ}.
 *
 * The array and the strings it lists are allocated using =malloc, and
 * should be released using =free after use:
 *
 *     char **envlist, **env;
 *
 *     envlist = environ;
 *     environ = pam_getenvlist(pamh);
 *     \/\* do something nifty \*\/
 *     for (env = environ; *env != NULL; env++)
 *         free(*env);
 *     free(environ);
 *     environ = envlist;
 *
 * >environ 7
 * >pam_getenv
 * >pam_putenv
 * >pam_setenv
 * >pam_unsetenv
 */
