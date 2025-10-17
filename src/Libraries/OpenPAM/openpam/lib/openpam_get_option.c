/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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

#include <string.h>

#include <security/pam_appl.h>
#include <security/openpam.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Returns the value of a module option
 */

const char *
openpam_get_option(pam_handle_t *pamh,
	const char *option)
{
	pam_chain_t *cur;
	size_t len;
	int i;

	ENTERS(option);
	if (pamh == NULL || pamh->current == NULL || option == NULL)
		RETURNS(NULL);
	cur = pamh->current;
	len = strlen(option);
	for (i = 0; i < cur->optc; ++i) {
		if (strncmp(cur->optv[i], option, len) == 0) {
			if (cur->optv[i][len] == '\0')
				RETURNS(&cur->optv[i][len]);
			else if (cur->optv[i][len] == '=')
				RETURNS(&cur->optv[i][len + 1]);
		}
	}
	RETURNS(NULL);
}

/**
 * The =openpam_get_option function returns the value of the specified
 * option in the context of the currently executing service module, or
 * =NULL if the option is not set or no module is currently executing.
 *
 * >openpam_set_option
 */
