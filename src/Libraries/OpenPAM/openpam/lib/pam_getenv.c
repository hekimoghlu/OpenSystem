/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
 * XSSO 6 page 44
 *
 * Retrieve the value of a PAM environment variable
 */

const char *
pam_getenv(pam_handle_t *pamh,
	const char *name)
{
	char *str;
	int i;

	ENTERS(name);
	if (pamh == NULL)
		RETURNS(NULL);
	if (name == NULL || strchr(name, '=') != NULL)
		RETURNS(NULL);
	if ((i = openpam_findenv(pamh, name, strlen(name))) < 0)
		RETURNS(NULL);
	for (str = pamh->env[i]; *str != '\0'; ++str) {
		if (*str == '=') {
			++str;
			break;
		}
	}
	RETURNS(str);
}

/**
 * The =pam_getenv function returns the value of an environment variable.
 * Its semantics are similar to those of =getenv, but it accesses the PAM
 * context's environment list instead of the application's.
 *
 * >pam_getenvlist
 * >pam_putenv
 * >pam_setenv
 * >pam_unsetenv
 */
