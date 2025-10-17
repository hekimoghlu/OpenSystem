/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>
#include <security/openpam.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Sets the value of a module option
 */

int
openpam_set_option(pam_handle_t *pamh,
	const char *option,
	const char *value)
{
	pam_chain_t *cur;
	char *opt, **optv;
	size_t len;
	int i;

	ENTERS(option);
	if (pamh == NULL || pamh->current == NULL || option == NULL)
		RETURNC(PAM_SYSTEM_ERR);
	cur = pamh->current;
	for (len = 0; option[len] != '\0'; ++len)
		if (option[len] == '=')
			break;
	for (i = 0; i < cur->optc; ++i) {
		if (strncmp(cur->optv[i], option, len) == 0 &&
		    (cur->optv[i][len] == '\0' || cur->optv[i][len] == '='))
			break;
	}
	if (value == NULL) {
		/* remove */
		if (i == cur->optc)
			RETURNC(PAM_SUCCESS);
		for (free(cur->optv[i]); i < cur->optc; ++i)
			cur->optv[i] = cur->optv[i + 1];
		cur->optv[i] = NULL;
		RETURNC(PAM_SUCCESS);
	}
	if (asprintf(&opt, "%.*s=%s", (int)len, option, value) < 0)
		RETURNC(PAM_BUF_ERR);
	if (i == cur->optc) {
		/* add */
		optv = realloc(cur->optv, sizeof(char *) * (cur->optc + 2));
		if (optv == NULL) {
			FREE(opt);
			RETURNC(PAM_BUF_ERR);
		}
		optv[i] = opt;
		optv[i + 1] = NULL;
		cur->optv = optv;
		++cur->optc;
	} else {
		/* replace */
		FREE(cur->optv[i]);
		cur->optv[i] = opt;
	}
	RETURNC(PAM_SUCCESS);
}

/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 */

/**
 * The =openpam_set_option function sets the specified option in the
 * context of the currently executing service module.
 *
 * >openpam_get_option
 */
